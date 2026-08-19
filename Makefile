PYTHONPATH := $(shell pwd)
export PYTHONPATH

# 반복 실험용 기본 하이퍼파라미터
ROUTE_RATIOS := 0.1 0.3 0.5
NODE_RATIOS  := 0.3 0.5 0.7

# 기본값 (make train seed=... 처럼 개별 덮어쓰기 가능)
model ?= lstm_gcn
route ?= 0.5
node  ?= 0.5
seed  ?= 42
gpu   ?= 0

.PHONY: help cluster inject inject_all train run tables clean-pyc

help:
	@echo "make cluster                                  # AIS -> 클러스터 원본 그래프 생성 (최초 1회)"
	@echo "make inject route=0.5 node=0.5                # 이상치 주입 그래프 생성"
	@echo "make inject_all                               # route/node 9개 조합 전부 주입"
	@echo "make train model=lstm_gcn route=0.5 node=0.5 seed=42"
	@echo "make run model=lstm_gcn gpu=0                 # 한 모델의 전체 조합 x 시드 반복 실험"
	@echo "make tables                                   # 모델별 결과를 지표별 비교표로 집계"

cluster:
	python scripts/cluster_once.py

inject:
	python scripts/inject_graphs.py $(route) $(node)

inject_all:
	@for r in $(ROUTE_RATIOS); do \
		for n in $(NODE_RATIOS); do \
			echo "==> make inject route=$$r node=$$n"; \
			$(MAKE) inject route=$$r node=$$n || echo "FAILED: route=$$r node=$$n"; \
		done \
	done

train:
	python scripts/train.py \
		--model $(model) \
		--route_ratio $(route) \
		--node_ratio $(node) \
		--seed $(seed)

run:
	bash scripts/run_experiments.sh $(model) $(gpu)

tables:
	python tools/make_comparison_tables.py

clean-pyc:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
