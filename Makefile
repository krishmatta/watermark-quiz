VENV := venv/bin

.PHONY: gen verify deploy-worker

gen:
	$(VENV)/modal run scripts/generate_modal.py

verify:
	$(VENV)/python scripts/verify.py

deploy-worker:
	cd worker && npx wrangler deploy
