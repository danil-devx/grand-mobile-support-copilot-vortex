PYTHON ?= python3

.PHONY: predict server test

predict:
	$(PYTHON) predict.py --in $(IN) --out $(OUT)

server:
	$(PYTHON) app.py

test:
	$(PYTHON) test_system.py
