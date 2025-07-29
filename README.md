# AI Character with Ollama, PostgreSQL, and Gradio

This project sets up a local AI Character using [Ollama](https://ollama.com/), a PostgreSQL database, and a Gradio interface.

## 1. Install and Run Ollama

### 1.1 Install Ollama

- **Linux**  
  Run the following command in your terminal:
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

- **Windows**  
  Download the installer from:  
  [https://ollama.com/download/windows](https://ollama.com/download/windows)

### 1.2 Start the Ollama Server

Open a new terminal and run:
```bash
ollama serve
```

### 1.3 Download the Model

Run this command in your terminal:
```bash
ollama pull mix_77/gemma3-qat-tools:12b
```

---

## 2. Install PostgreSQL (via Docker)

Start PostgreSQL using Docker:
```bash
docker run -d -e POSTGRES_PASSWORD=abc123 -p 5432:5432 postgres
```

Check if PostgreSQL is running:
```bash
pg_isready
```

If successful, you should see:
```
/var/run/postgresql:5432 - accepting connections
```

---

## 3. Set Up Python Environment

Install dependencies using pip:
```bash
pip install -r requirement.txt
```

---

## 4. Configure Environment Variables

Create the `.env` file:
```bash
cp .env.template .env
```

Edit `.env` and fill in the required fields:

### Required:
```env
OPENAI_API_KEY=sk-F3M...
POSTGRES_URI=postgresql://postgres:admin@localhost:5432/postgres?sslmode=disable
HF_TOKEN=hf_FXb...
```

### Optional:
```env
LANGCHAIN_TRACING_V2=true
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=lsv2....
LANGCHAIN_PROJECT=AI-Character-Chat-Dev
```

---

## . Run the Gradio App

Set the Python path and run the main file:
```bash
export PYTHONPATH="./"
python src/main.py
```

---