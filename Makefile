VENV := venv/bin

.PHONY: gen verify sync deploy-worker

gen:
	$(VENV)/modal run scripts/generate_modal.py

verify:
	$(VENV)/python scripts/verify.py

sync:
	cp site/index.html site/pairs.json ~/org/static/watermark-quiz/

deploy-worker:
	cd worker && npx wrangler deploy
