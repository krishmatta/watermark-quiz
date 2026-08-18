.PHONY: gen verify deploy-worker

gen:
	uv run modal run scripts/generate_modal.py

verify:
	uv run scripts/verify.py

deploy-worker:
	cd worker && npx wrangler deploy
