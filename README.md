# 🧠 Complete Formation - Artificial Intelligence

Este repositório contém o material completo da formação em Inteligência Artificial, utilizando Python com ambiente virtual (`venv`), integração com o JupyterLab no VSCode e versionamento com arquivos `.py` estruturados como notebooks.

---

## ✅ Objetivo

Proporcionar um ambiente de estudo eficiente e versionável em IA, com execução prática via células interativas (`# %%`) no VSCode.

---

## ⚙️ Requisitos

Antes de começar, instale:

- Python 3.13+
- [VSCode](https://code.visualstudio.com/)
- Extensão **Jupyter** **Python** da microsoft no VSCode
- Git
- `pip`

---

## 🛠️ Configurando o ambiente virtual e instalando dependências

### 1. Crie o ambiente virtual

No terminal, dentro da pasta do projeto, execute:

```bash
python -m venv .venv
```

### 2. Ative o ambiente virtual
#### Windows 
```
.venv\Scripts\activate
```

#### Linux/macOS: 
```
source .venv/bin/activate
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```
---
## 🧪 Como executar os arquivos como notebooks

Este projeto utiliza arquivos `.py` com marcações de células (`# %%`). Isso permite executá-los como notebooks diretamente no VSCode.

### Como usar:

- Abra o projeto no VSCode.
- Certifique-se de que a extensão **Jupyter (Microsoft)** está instalada.
- Abra um arquivo `.py` (ex: `001_intro_ai.py`).
- Você verá divisões por células marcadas com `# %%`.
- Para executar uma célula, clique sobre ela e pressione:
  - `Ctrl+Enter` para rodar a célula
  - `Shift+Enter` para rodar e ir para a próxima

### Como criar uma nova célula:

Basta adicionar o marcador:

```python
# %%
