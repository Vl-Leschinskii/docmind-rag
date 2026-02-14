from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import shutil
from pathlib import Path
from orchestrator import RAGOrchestrator
import traceback

app = FastAPI(title="DocMind Local RAG")

# Создаем директории
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ГЛОБАЛЬНЫЙ оркестратор - ОДИН для всех запросов!
print("🔄 Инициализация глобального оркестратора...")
orchestrator = RAGOrchestrator("config.yaml")
print(f"✅ Оркестратор инициализирован: {orchestrator}")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Главная страница"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>DocMind Local RAG</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                margin-bottom: 20px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .upload-area {
                border: 2px dashed #667eea;
                border-radius: 10px;
                padding: 40px;
                text-align: center;
                background: #f8f9ff;
                cursor: pointer;
                transition: all 0.3s;
            }
            .upload-area:hover {
                background: #eef2ff;
                border-color: #764ba2;
            }
            .upload-area input {
                display: none;
            }
            .upload-label {
                cursor: pointer;
            }
            .upload-icon {
                font-size: 48px;
                color: #667eea;
                margin-bottom: 10px;
            }
            .progress-bar {
                width: 100%;
                height: 6px;
                background: #f0f0f0;
                border-radius: 3px;
                margin-top: 20px;
                display: none;
            }
            .progress-fill {
                width: 0%;
                height: 100%;
                background: linear-gradient(90deg, #667eea, #764ba2);
                border-radius: 3px;
                transition: width 0.3s;
            }
            .document-info {
                margin-top: 30px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
                display: none;
            }
            .query-section {
                margin-top: 30px;
                display: none;
            }
            .query-input {
                display: flex;
                gap: 10px;
            }
            input[type="text"] {
                flex: 1;
                padding: 12px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 16px;
                transition: border-color 0.3s;
            }
            input[type="text"]:focus {
                border-color: #667eea;
                outline: none;
            }
            button {
                padding: 12px 24px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                cursor: pointer;
                transition: transform 0.2s;
            }
            button:hover {
                transform: translateY(-2px);
            }
            .result {
                margin-top: 30px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
                display: none;
            }
            .answer {
                font-size: 16px;
                line-height: 1.6;
                color: #333;
            }
            .sources {
                margin-top: 20px;
                padding-top: 20px;
                border-top: 1px solid #e0e0e0;
            }
            .source-item {
                background: white;
                padding: 10px;
                margin: 5px 0;
                border-radius: 5px;
                font-size: 14px;
                color: #666;
            }
            .warning {
                background: #fff3cd;
                color: #856404;
                padding: 10px;
                border-radius: 5px;
                margin-top: 10px;
            }
            .confidence {
                display: inline-block;
                padding: 4px 8px;
                border-radius: 4px;
                background: #d4edda;
                color: #155724;
                font-size: 14px;
            }
            .structure-tree {
                margin-top: 20px;
                padding: 15px;
                background: white;
                border-radius: 8px;
                max-height: 300px;
                overflow-y: auto;
            }
            .tree-item {
                padding: 5px 0 5px 20px;
                border-left: 2px solid #667eea;
                margin: 5px 0;
            }
            .tree-item.chapter {
                border-left-color: #764ba2;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📚 DocMind Local RAG</h1>
            <p>Мультиагентная система для анализа больших Word документов</p>
            
            <div class="upload-area" onclick="document.getElementById('file-input').click()">
                <input type="file" id="file-input" accept=".docx">
                <div class="upload-icon">📄</div>
                <h3>Загрузите Word документ</h3>
                <p>Поддерживаются файлы .docx до 100+ страниц</p>
            </div>
            
            <div class="progress-bar" id="progress-bar">
                <div class="progress-fill" id="progress-fill"></div>
            </div>
            
            <div class="document-info" id="document-info">
                <h2>📊 Информация о документе</h2>
                <p id="doc-name"></p>
                <p id="doc-stats"></p>
                <div id="structure-tree" class="structure-tree"></div>
            </div>
            
            <div class="query-section" id="query-section">
                <h2>❓ Задайте вопрос по документу</h2>
                <div class="query-input">
                    <input type="text" id="query-input" placeholder="Введите ваш вопрос...">
                    <button onclick="askQuestion()">Отправить</button>
                </div>
            </div>
            
            <div class="result" id="result">
                <h2>📝 Ответ</h2>
                <div class="answer" id="answer"></div>
                <div id="confidence"></div>
                <div id="warnings"></div>
                <div class="sources" id="sources"></div>
            </div>
        </div>

        <script>
            let isDocumentLoaded = false;
            
            document.getElementById('file-input').addEventListener('change', async function(e) {
                const file = e.target.files[0];
                if (!file) return;
                
                const formData = new FormData();
                formData.append('file', file);
                
                const progressBar = document.getElementById('progress-bar');
                const progressFill = document.getElementById('progress-fill');
                progressBar.style.display = 'block';
                progressFill.style.width = '50%';
                
                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    progressFill.style.width = '100%';
                    
                    if (response.ok) {
                        const data = await response.json();
                        setTimeout(() => {
                            progressBar.style.display = 'none';
                            progressFill.style.width = '0%';
                            showDocumentInfo(data);
                        }, 500);
                    }
                } catch (error) {
                    alert('Ошибка загрузки файла');
                    progressBar.style.display = 'none';
                }
            });
            
            function showDocumentInfo(data) {
                document.getElementById('doc-name').innerHTML = `<strong>Файл:</strong> ${data.filename}`;
                document.getElementById('doc-stats').innerHTML = `<strong>Глав:</strong> ${data.chapters} | <strong>Чанков:</strong> ${data.chunks}`;
                
                // Отображаем структуру
                const structure = data.structure;
                let treeHtml = '<h4>📑 Структура документа:</h4>';
                
                structure.chapters.forEach(chapter => {
                    treeHtml += `<div class="tree-item chapter">📖 ${chapter.title}</div>`;
                    chapter.sections.forEach(section => {
                        treeHtml += `<div class="tree-item">📌 ${section.title}</div>`;
                    });
                });
                
                document.getElementById('structure-tree').innerHTML = treeHtml;
                document.getElementById('document-info').style.display = 'block';
                document.getElementById('query-section').style.display = 'block';
                isDocumentLoaded = true;
            }
            
            async function askQuestion() {
                const query = document.getElementById('query-input').value;
                if (!query) {
                    alert('Введите вопрос');
                    return;
                }
                
                const resultDiv = document.getElementById('result');
                const answerDiv = document.getElementById('answer');
                const confidenceDiv = document.getElementById('confidence');
                const warningsDiv = document.getElementById('warnings');
                const sourcesDiv = document.getElementById('sources');
                
                answerDiv.innerHTML = '🤔 Генерация ответа...';
                resultDiv.style.display = 'block';
                
                try {
                    const response = await fetch(`/query?q=${encodeURIComponent(query)}`);
                    const data = await response.json();
                    
                    answerDiv.innerHTML = data.answer.replace(/\\n/g, '<br>');
                    
                    // Уверенность
                    const confidencePercent = Math.round(data.confidence * 100);
                    let confidenceClass = 'confidence';
                    let confidenceText = `Уверенность: ${confidencePercent}%`;
                    confidenceDiv.innerHTML = `<span class="${confidenceClass}">${confidenceText}</span>`;
                    
                    // Предупреждения
                    if (data.warnings && data.warnings.length > 0) {
                        let warningsHtml = '';
                        data.warnings.forEach(warning => {
                            warningsHtml += `<div class="warning">${warning}</div>`;
                        });
                        warningsDiv.innerHTML = warningsHtml;
                    } else {
                        warningsDiv.innerHTML = '';
                    }
                    
                    // Источники
                    if (data.sources && data.sources.length > 0) {
                        let sourcesHtml = '<h4>📚 Источники:</h4>';
                        data.sources.forEach(source => {
                            sourcesHtml += `<div class="source-item">📖 ${source.chapter} → 📌 ${source.section}</div>`;
                        });
                        sourcesDiv.innerHTML = sourcesHtml;
                    }
                    
                } catch (error) {
                    answerDiv.innerHTML = '❌ Ошибка при получении ответа';
                }
            }
        </script>
    </body>
    </html>
    """

# web_interface.py - обновленная функция upload_document

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Загрузка и обработка документа"""
    print(f"\n📥 ПОЛУЧЕН ЗАПРОС НА ЗАГРУЗКУ: {file.filename}")
    
    if not file.filename.endswith('.docx'):
        print("❌ Неверный формат файла")
        return JSONResponse(
            status_code=400,
            content={"error": "Только .docx файлы поддерживаются"}
        )
    
    # Сохраняем файл
    file_path = UPLOAD_DIR / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"✅ Файл сохранен: {file_path}")
    except Exception as e:
        print(f"❌ Ошибка сохранения файла: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Ошибка сохранения файла: {str(e)}"}
        )
    
    # Обрабатываем документ через ГЛОБАЛЬНЫЙ оркестратор
    try:
        print(f"🔄 Вызов orchestrator.process_document()")
        print(f"   orchestrator: {orchestrator}")
        print(f"   Доступные методы: {dir(orchestrator)}")
        
        # Проверяем наличие метода
        if not hasattr(orchestrator, 'process_document'):
            print("❌ У orchestrator нет метода process_document!")
            return JSONResponse(
                status_code=500,
                content={"error": "Внутренняя ошибка сервера: отсутствует метод process_document"}
            )
        
        result = orchestrator.process_document(str(file_path))
        
        print(f"✅ Документ обработан успешно!")
        print(f"   Глав: {result['chapters_count']}")
        print(f"   Чанков: {result['chunks_count']}")
        
        return {
            "status": "success",
            "filename": file.filename,
            "chapters": result['chapters_count'],
            "chunks": result['chunks_count'],
            "structure": result['structure']
        }
    except Exception as e:
        print(f"❌ ОШИБКА при обработке документа:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )

@app.get("/query")
async def query(q: str):
    """Запрос к документу"""
    print(f"\n❓ ПОЛУЧЕН ЗАПРОС: {q}")
    
    try:
        result = orchestrator.query_document(q)
        print(f"✅ Ответ сгенерирован. Уверенность: {result.get('confidence', 0)}")
        return result
    except Exception as e:
        print(f"❌ Ошибка при генерации ответа: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/structure")
async def get_structure():
    """Получение структуры документа"""
    structure = orchestrator.get_document_structure()
    return structure

@app.get("/debug")
async def debug():
    """Отладочная информация"""
    return {
        "orchestrator_exists": orchestrator is not None,
        "is_indexed": orchestrator.is_indexed if orchestrator else False,
        "doc_structure": orchestrator.doc_structure is not None
    }

def start_server():
    """Запуск веб-сервера"""
    print("🚀 Запуск веб-сервера...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

if __name__ == "__main__":
    start_server()