PY ?= python
export PYTHONPATH := $(CURDIR)

.PHONY: install lock build prepare train index evaluate llm-eval feedback gate report test lint api api-dev web-install web-dev web-build demo
install:      ; $(PY) -m pip install -r requirements.lock
lock:         ; sh scripts/lock_requirements.sh $(PY)
build:        ; $(PY) -m ssa.cli build && $(PY) -m ssa.cli llm-eval && $(PY) -m ssa.cli report
prepare:      ; $(PY) -m ssa.cli prepare
train:        ; $(PY) -m ssa.cli train
index:        ; $(PY) -m ssa.cli index
evaluate:     ; $(PY) -m ssa.cli evaluate
llm-eval:     ; $(PY) -m ssa.cli llm-eval
feedback:     ; $(PY) -m ssa.cli feedback-loop
gate:         ; $(PY) -m ssa.cli quality-gate --with-tests
report:       ; $(PY) scripts/render_architecture.py && $(PY) -m ssa.cli report
test:         ; $(PY) -m pytest
lint:         ; ruff check ssa tests scripts
api:          ; uvicorn ssa.api.app:app_factory --factory --host 127.0.0.1 --port 8000
api-dev:      ; SSA_DEV_MODE=1 uvicorn ssa.api.app:app_factory --factory --reload --port 8000
web-install:  ; cd frontend && npm install
web-dev:      ; cd frontend && npm run dev
web-build:    ; cd frontend && npm run build
demo:         ; SSA_HOME=$(CURDIR)/.demo $(PY) -m ssa.cli demo-data && SSA_HOME=$(CURDIR)/.demo $(PY) -m ssa.cli build --no-dense
