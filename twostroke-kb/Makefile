.PHONY: setup db run test fmt clean

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
	@echo ""
	@echo "System deps still needed (not pip):"
	@echo "  - tesseract-ocr + tesseract-ocr-deu   (OCR)"
	@echo "  - poppler-utils                        (pdf2image)"
	@echo "  - libreoffice OR antiword              (legacy .doc)"
	@echo ""
	@echo "Then: cp .env.example .env  and fill it in."

db:
	docker compose up -d
	@echo "Postgres+pgvector running on localhost:5432 (schema auto-applied on first run)."

run:
	. .venv/bin/activate && uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

test:
	. .venv/bin/activate && pytest -q

clean:
	rm -rf __pycache__ .pytest_cache **/__pycache__
