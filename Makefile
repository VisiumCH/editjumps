# Common workflows. `make` or `make help` lists targets.
.PHONY: alignment-scoring check ci consolidate deterministic-benchmark disjoint-baselines disjoint-editor dvc-pull dvc-push eval-sweep evodiff-baseline evotune evotune-baselines fetch-repro figure-extract generation-eval help hero install-evodiff install-mmseqs jobs-calibrate jobs-cancel jobs-experiment jobs-geneval jobs-logs jobs-lr-sweep jobs-repro jobs-status jobs-train jobs-train-faithful mlflow-ui provenance repro require-sky restore-editor test

# Default parameters
EDITOR ?= data/pretrain/eval_B_stock_appA
STAGES ?= pretrain_esm
TRUNKS ?= data/local_dj
REPRO_STAGE_DIR ?= data/interim/repro-artifacts
DISJOINT_FRAME = --disjoint-from-pairs data/pretrain/oas_homolog_pairs.tsv.gz \
                 --n-templates 20 --n-variants 20 --holdout-size 200 --ceiling-n 300 --seed 0 \
                 --budget-mode realised

help:  ## list targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}'

# --- Development & Quality ---
check:  ## lint + type-check (ruff + ty)
	uv run --locked ruff check . && uv run --locked ty check .

test:  ## run the test suite
	uv run --locked pytest -q

ci:  ## run the fast local CI checks (ruff, ty, pytest) -- torch tests SKIP here, see ci-train
	uv sync --locked --all-extras --dev
	uv run --locked ruff check .
	uv run --locked ty check .
	uv run --locked pytest

ci-train:  ## the second CI lane: ty + pytest with the train group, so the torch tests actually run
	uv sync --locked --all-extras --all-groups
	uv run --locked ty check .
	uv run --locked pytest

provenance:  ## regenerate docs/claims.md table from editjumps/core/provenance.py
	uv run editjumps provenance --write

hero:  ## regenerate README hero SVGs (light + dark)
	uv run python editjumps/repo/make_hero.py

# --- Data & Setup ---
install-mmseqs:  ## install MMseqs2 binary into .venv/bin (required by `make repro`)
	bash editjumps/core/install_mmseqs.sh

install-evodiff:  ## install EvoDiff (+MSA weights) into an isolated env (for evodiff-baseline)
	bash editjumps/core/evodiff_msa/install_evodiff.sh

repro:  ## dvc repro of $(STAGES) (default: pretrain_esm; STAGES= for full DAG)
	uv run dvc repro $(STAGES)

require-dvc-bucket:
	@test -n "$(DVC_BUCKET)" || { \
	  echo "ERROR: DVC_BUCKET environment variable is not set."; \
	  echo "Please export it in your shell: export DVC_BUCKET=gs://<your-bucket-name>"; \
	  exit 1; }

ensure-dvc-remote: require-dvc-bucket
	@uv run dvc remote modify --local gcs url "$$(echo $(DVC_BUCKET) | sed 's|/*$$||')/dvc" >/dev/null

dvc-push: ensure-dvc-remote  ## push DVC cache to GCS remote
	uv run dvc push

dvc-pull: ensure-dvc-remote  ## pull DVC-tracked data from GCS remote
	uv run dvc pull

mlflow-ui:  ## local MLflow UI against sqlite:///mlflow.db (override PORT=; default 5555)
	bash editjumps/repo/mlflow_ui.sh --port $(or $(PORT),5555)

# --- Reproduction & Evaluation ---
consolidate:  ## collect every §4.2 result into metrics/all_results.{json,csv} and appendix_table.md
	uv run editjumps consolidate --out-md metrics/appendix_table.md

figure-extract:  ## recover EvoFlows Figure 3 data points from PDF (PDF= PAGE=)
	uv run --with pymupdf editjumps figure-extract \
	  --pdf $(or $(PDF),paper.pdf) --page-number $(or $(PAGE),9) \
	  --output $(or $(OUT),metrics/evoflows_figure3.json)

eval-sweep:  ## run {arm} x {clock} x {family} generation sweep -> metrics/sweep/
	bash editjumps/measurements/eval_sweep.sh

generation-eval:  ## EvoFlows Appendix-B metrics for a trained editor (MODEL= N_TEMPLATES= N_VARIANTS= HOLDOUT_SIZE=)
	uv run --group train editjumps generation-eval \
	  --model-folder $(or $(MODEL),data/pretrain/edit_flows_baseline) \
	  $(if $(N_TEMPLATES),--n-templates $(N_TEMPLATES),) \
	  $(if $(N_VARIANTS),--n-variants $(N_VARIANTS),) \
	  $(if $(HOLDOUT_SIZE),--holdout-size $(HOLDOUT_SIZE),) \
	  $(if $(RATE_HEAD),--rate-head $(RATE_HEAD),) $(if $(Q_HEAD),--q-head $(Q_HEAD),) \
	  $(if $(CLOCK),--clock $(CLOCK),) $(if $(PLL_MODEL),--pll-model $(PLL_MODEL),) \
	  $(if $(FAMILY),--family-fasta $(FAMILY),) \
	  --metrics-path $(or $(OUT),metrics/generation_eval.json)

alignment-scoring:  ## measure alignment scoring effect on edit-op label distribution (N=)
	uv run editjumps alignment-scoring $(if $(N),--n-pairs $(N),)

deterministic-benchmark:  ## EvoFlows §4.1: build synthetic pairs, train, score per-class P/R
	uv run dvc repro build_deterministic_pairs train_deterministic_editor deterministic_benchmark

evotune:  ## §2.2 Evotuning: MLM-adapt ESM-2 to one seed family (FAMILY= MODEL= EPOCHS=)
	uv run --group train editjumps evotune \
	  $(if $(FAMILY),--family-fasta $(FAMILY),) \
	  $(if $(MODEL),--model-name $(MODEL),) \
	  $(if $(EPOCHS),--epochs $(EPOCHS),) \
	  --metrics-path $(or $(OUT),metrics/evotune_esm.json)

evotune-baselines:  ## EvoFlows evotuned-PLM baselines, forced and unforced (MUTATIONS= FROM= TEMPERATURE= PLL_MODEL=)
	@for forced in --no-forced --forced; do \
	  out=metrics/evotune_baseline$$([ "$$forced" = "--forced" ] && echo _forced).json; \
	  uv run --group train editjumps evotune-baseline $$forced \
	    $(if $(FAMILY),--family-fasta $(FAMILY),) \
	    $(if $(MUTATIONS),--mutations $(MUTATIONS),) \
	    $(if $(FROM),--mutations-from $(FROM),) \
	    $(if $(TEMPERATURE),--temperature $(TEMPERATURE),) \
	    $(if $(PLL_MODEL),--pll-model $(PLL_MODEL),) \
	    --metrics-path $$out || exit 1; \
	done

evodiff-baseline:  ## EvoFlows EvoDiff-MSA baseline (MODE= MUTATIONS= FROM= MODEL= DEVICE= PLL_MODEL=)
	uv run editjumps evodiff-msa-baseline \
	  $(if $(FAMILY),--family-fasta $(FAMILY),) \
	  $(if $(MODE),--mode $(MODE),) \
	  $(if $(MODEL),--model $(MODEL),) \
	  $(if $(MUTATIONS),--mutations $(MUTATIONS),) \
	  $(if $(FROM),--mutations-from $(FROM),) \
	  $(if $(DEVICE),--device $(DEVICE),) \
	  $(if $(PLL_MODEL),--pll-model $(PLL_MODEL),) \
	  --metrics-path $(or $(OUT),metrics/evodiff_msa_baseline.json)

disjoint-editor:  ## the editor's 2 §4.2 cells + 2 model-free baselines (EDITOR=)
	@for fam in ty1:Anti-SARS-CoV-2_VHH_Ty1 her2vh:Anti-HER2_scFv_VH_trastuzumab; do \
	  tag=$${fam%%:*}; name=$${fam##*:}; \
	  family=data/interim/seed_families/$$name.fasta; \
	  test -f $$family || { echo "missing $$family - build it with 'dvc pull' or dvc repro seed_homologs"; exit 1; }; \
	  uv run --group train editjumps generation-eval \
	    --model-folder $(EDITOR) --rate-head mlp --q-head esm_lm_head \
	    --family-fasta $$family \
	    --pairs data/pretrain/oas_homolog_pairs.tsv.gz --disjoint-from-pairs \
	    --n-templates 20 --n-variants 20 --holdout-size 200 --ceiling-n 300 \
	    --seed 0 --n-steps 50 --clock 40 \
	    --metrics-path metrics/disjoint/editor-$$tag.json || exit 1; \
	done

disjoint-baselines:  ## the 6 model-based §4.2 cells on disjoint reference (TRUNKS=)
	@for fam in ty1:Anti-SARS-CoV-2_VHH_Ty1 her2vh:Anti-HER2_scFv_VH_trastuzumab; do \
	  tag=$${fam%%:*}; name=$${fam##*:}; \
	  family=data/interim/seed_families/$$name.fasta; \
	  editor=metrics/disjoint/editor-$$tag.json; \
	  test -f $$editor || (echo "missing $$editor: the mutation budget is matched to it"; exit 1); \
	  uv run editjumps evodiff-msa-baseline --family-fasta $$family $(DISJOINT_FRAME) \
	    --mutations-from $$editor --mode inpaint --device cpu \
	    --metrics-path metrics/disjoint/evodiff-msa-$$tag.json || exit 1; \
	  for forced in --no-forced --forced; do \
	    out=metrics/disjoint/evotune$$([ "$$forced" = "--forced" ] && echo -forced)-$$tag.json; \
	    uv run --group train editjumps evotune-baseline $$forced --family-fasta $$family \
	      --model-folder $(TRUNKS)/evotune-dj-$$tag \
	      --train-corpus $(TRUNKS)/evotune-dj-$$tag/evotune_family.train.txt.gz \
	      $(DISJOINT_FRAME) --mutations-from $$editor --metrics-path $$out || exit 1; \
	  done; \
	done

restore-editor:  ## rebuild loadable editor folder from checkpoint.pt (CKPT= OUT=)
	@test -n "$(CKPT)" || (echo "CKPT= is required (local checkpoint.pt or gs:// URI)"; exit 1)
	uv run --group train editjumps restore-editor --checkpoint "$(CKPT)" \
	  --output-folder $(or $(OUT),data/pretrain/edit_flows_restored) \
	  $(if $(MODEL_NAME),--model-name $(MODEL_NAME),) \
	  $(if $(RATE_HEAD),--rate-head $(RATE_HEAD),) $(if $(Q_HEAD),--q-head $(Q_HEAD),)

# --- Cloud Execution (SkyPilot) ---
require-sky:
	@command -v sky >/dev/null || { \
	  echo "sky not found - install SkyPilot CLI: uv tool install 'skypilot[gcp]' && sky check gcp"; \
	  exit 1; }

jobs-repro: require-sky require-dvc-bucket  ## managed SPOT repro job (STAGES= GPU=)
	sky jobs launch -n editjumps-repro --use-spot -y deploy/gcp/repro.sky.yaml \
	  $(if $(GPU),--gpus $(GPU),) --env STAGES="$(STAGES)" \
	  --env DVC_BUCKET="$(DVC_BUCKET)"

fetch-repro: require-dvc-bucket  ## collect dvc.lock + metrics from cloud job, diff lock (RUN_TAG=)
	@mkdir -p $(REPRO_STAGE_DIR)
	gcloud storage cp $(DVC_BUCKET)/repro-artifacts/dvc.lock $(REPRO_STAGE_DIR)/dvc.lock
	-gcloud storage cp "$(DVC_BUCKET)/repro-artifacts/$(if $(RUN_TAG),$(RUN_TAG)-,)*.json" $(REPRO_STAGE_DIR)/
	@echo "--- dvc.lock: local vs the job's (nothing overwritten) ---"
	@diff dvc.lock $(REPRO_STAGE_DIR)/dvc.lock && echo "identical - nothing to adopt" || \
	  echo "^ review the stage(s) above, then: cp $(REPRO_STAGE_DIR)/dvc.lock dvc.lock && uv run dvc pull"

jobs-train: require-sky require-dvc-bucket  ## managed SPOT edit-flow training (RUN_TAG= MODEL_NAME= RATE_HEAD= Q_HEAD= SCHEDULE= GPU= REGION=)
	sky jobs launch -n editjumps-$(or $(RUN_TAG),train) --use-spot -y deploy/gcp/train.sky.yaml \
	  $(if $(GPU),--gpus $(GPU),) $(if $(REGION),--region $(REGION),) \
	  --env DVC_BUCKET="$(DVC_BUCKET)" \
	  $(if $(RUN_TAG),--env RUN_TAG=$(RUN_TAG),) \
	  $(if $(MODEL_NAME),--env MODEL_NAME=$(MODEL_NAME),) \
	  $(if $(RATE_HEAD),--env RATE_HEAD=$(RATE_HEAD),) \
	  $(if $(Q_HEAD),--env Q_HEAD=$(Q_HEAD),) \
	  $(if $(SCHEDULE),--env SCHEDULE=$(SCHEDULE),)

jobs-train-faithful:  ## editor run matching paper choices (stock ESM-2 + Appendix-A heads)
	$(MAKE) jobs-train RUN_TAG=$(or $(RUN_TAG),faithful-appendixa) \
	  MODEL_NAME=$(or $(MODEL_NAME),facebook/esm2_t12_35M_UR50D) RATE_HEAD=mlp Q_HEAD=esm_lm_head

jobs-experiment: require-sky require-dvc-bucket  ## managed SPOT hyperparameter run (RUN_TAG= LR= BATCH_SIZE= MAX_STEPS= MODEL_NAME= RATE_HEAD= Q_HEAD= GPU= REGION=)
	@test -n "$(RUN_TAG)" || (echo "RUN_TAG= is required (keys checkpoint prefix)"; exit 1)
	sky jobs launch -n editjumps-exp-$(RUN_TAG) --use-spot -y $(if $(ASYNC),--async,) deploy/gcp/train.sky.yaml \
	  $(if $(GPU),--gpus $(GPU),) $(if $(REGION),--region $(REGION),) \
	  --env DVC_BUCKET="$(DVC_BUCKET)" \
	  --env RUN_TAG="$(RUN_TAG)" --env LR="$(or $(LR),1e-4)" \
	  --env BATCH_SIZE="$(or $(BATCH_SIZE),16)" --env MAX_STEPS="$(or $(MAX_STEPS),20000)" \
	  $(if $(MODEL_NAME),--env MODEL_NAME=$(MODEL_NAME),) \
	  $(if $(RATE_HEAD),--env RATE_HEAD=$(RATE_HEAD),) \
	  $(if $(Q_HEAD),--env Q_HEAD=$(Q_HEAD),)

jobs-lr-sweep:  ## round-1 LR sweep at 650M (LR_GRID= MAX_STEPS= GPU= REGION=)
	@echo "launching $(words $(LR_GRID)) short 650M runs at LR in: $(LR_GRID)"
	@for lr in $(LR_GRID); do \
	  echo "--- $$lr"; \
	  $(MAKE) --no-print-directory jobs-experiment \
	    RUN_TAG="lr650m-$$lr" LR="$$lr" \
	    MAX_STEPS="$(or $(MAX_STEPS),3000)" BATCH_SIZE="$(or $(BATCH_SIZE),16)" \
	    MODEL_NAME=facebook/esm2_t33_650M_UR50D RATE_HEAD=mlp Q_HEAD=esm_lm_head \
	    ASYNC="$(or $(ASYNC),1)" \
	    GPU="$(or $(GPU),A100:1)" REGION="$(or $(REGION),europe-west4)" || exit 1; \
	done
	@echo "queued. watch: make jobs-status ; then score checkpoints with make generation-eval"

jobs-calibrate: require-sky require-dvc-bucket  ## measure s/step + MFU across batch sizes on GPU (BATCH_SIZES=16,64,256)
	sky jobs launch -n editjumps-calibrate --use-spot -y deploy/gcp/train.sky.yaml \
	  --env DVC_BUCKET="$(DVC_BUCKET)" \
	  --env MODE=calibrate --env RUN_TAG=$(or $(RUN_TAG),calibrate) \
	  --env BATCH_SIZES="$(if $(BATCH_SIZES),$(BATCH_SIZES),16,64,256)"

jobs-geneval: require-sky require-dvc-bucket  ## GPU Appendix-B generation eval for one arm (RUN_TAG= MODEL_NAME= RATE_HEAD= Q_HEAD= FAMILY=)
	@test -n "$(RUN_TAG)" || (echo "RUN_TAG= is required"; exit 1)
	sky jobs launch -n editjumps-geneval-$(or $(METRICS_TAG),$(RUN_TAG)) --use-spot -y deploy/gcp/train.sky.yaml \
	  --env DVC_BUCKET="$(DVC_BUCKET)" \
	  --env MODE=geneval --env RUN_TAG=$(RUN_TAG) \
	  $(if $(MODEL_NAME),--env MODEL_NAME=$(MODEL_NAME),) \
	  $(if $(RATE_HEAD),--env RATE_HEAD=$(RATE_HEAD),) $(if $(Q_HEAD),--env Q_HEAD=$(Q_HEAD),) \
	  $(if $(FAMILY),--env FAMILY=$(FAMILY),) $(if $(PLL_MODEL),--env PLL_MODEL=$(PLL_MODEL),) \
	  $(if $(CLOCK),--clock $(CLOCK),) \
	  $(if $(N_TEMPLATES),--env N_TEMPLATES=$(N_TEMPLATES),) \
	  $(if $(N_VARIANTS),--env N_VARIANTS=$(N_VARIANTS),) \
	  $(if $(HOLDOUT_SIZE),--env HOLDOUT_SIZE=$(HOLDOUT_SIZE),) \
	  $(if $(METRICS_TAG),--env METRICS_TAG=$(METRICS_TAG),)

jobs-status: require-sky  ## managed-jobs queue status
	sky jobs queue

jobs-logs: require-sky  ## follow managed job logs (NAME=)
	sky jobs logs -n $(NAME)

jobs-cancel: require-sky  ## cancel managed job (NAME=)
	sky jobs cancel -n $(NAME) -y
