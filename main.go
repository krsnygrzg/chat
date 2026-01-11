package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/gorilla/mux"
)

type PromptRequest struct {
	Prompt      string  `json:"prompt"`
	MaxTokens   int     `json:"max_tokens,omitempty"`
	Temperature float64 `json:"temperature,omitempty"`
}

type OllamaGenerateRequest struct {
	Model   string   `json:"model"`
	Prompt  string   `json:"prompt"`
	Stream  bool     `json:"stream"`
	Options *Options `json:"options,omitempty"`
}

type Options struct {
	NumPredict  int     `json:"num_predict,omitempty"` // аналог max_tokens
	Temperature float64 `json:"temperature,omitempty"`
}

type OllamaGenerateResponse struct {
	Response string `json:"response"`
	// есть и другие поля, но нам достаточно response
}

type APIResponse struct {
	Generated string `json:"generated"`
}

func callOllama(model string, req PromptRequest) (string, error) {
	url := "http://localhost:11434/api/generate"

	ollamaReq := OllamaGenerateRequest{
		Model:  model,
		Prompt: req.Prompt,
		Stream: false, // важно: чтобы пришёл один JSON, а не поток
		Options: &Options{
			NumPredict:  req.MaxTokens,
			Temperature: req.Temperature,
		},
	}

	body, err := json.Marshal(ollamaReq)
	if err != nil {
		return "", err
	}

	httpClient := &http.Client{Timeout: 60 * time.Second}
	httpResp, err := httpClient.Post(url, "application/json", bytes.NewBuffer(body))
	if err != nil {
		return "", err
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode != http.StatusOK {
		respBytes, _ := io.ReadAll(httpResp.Body)
		return "", fmt.Errorf("ollama error: status=%d body=%s", httpResp.StatusCode, string(respBytes))
	}

	var ollamaResp OllamaGenerateResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&ollamaResp); err != nil {
		return "", err
	}

	return ollamaResp.Response, nil
}

func predictHandler(w http.ResponseWriter, r *http.Request) {
	var req PromptRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body: "+err.Error(), http.StatusBadRequest)
		return
	}
	if req.Prompt == "" {
		http.Error(w, "prompt is required", http.StatusBadRequest)
		return
	}

	modelName := "qwen3:4b"

	result, err := callOllama(modelName, req)
	if err != nil {
		http.Error(w, "model error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	resp := APIResponse{Generated: result}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func main() {
	r := mux.NewRouter()
	r.HandleFunc("/predict", predictHandler).Methods("POST")

	srv := &http.Server{
		Handler:      r,
		Addr:         "0.0.0.0:8080",
		WriteTimeout: 15 * time.Second,
		ReadTimeout:  15 * time.Second,
	}
	fmt.Println("Server listening on http://0.0.0.0:8080")
	if err := srv.ListenAndServe(); err != nil {
		fmt.Println("Server error:", err)
	}
}

// {
//   "model":"qwen3:4b",
//   "prompt":"Ответь одним абзацем. Без рассуждений. Только финальный ответ.\n\nРасскажи анекдот на тему Go.",
//   "stream":false,
//   "options":{"num_predict":200,"temperature":0.7}
// }
