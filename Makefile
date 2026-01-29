
build:
	docker build -t llama-edge-image .

run:
	docker run -d -p 8000:8000 --name llama-edge-container --env-file .env llama-edge-image