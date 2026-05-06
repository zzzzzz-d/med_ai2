FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MED_AI_PROJECT_ROOT=/app \
    MED_AI_FEEDBACK_DIR=/data/feedback_data \
    MED_AI_METRIC_STORE_PATH=/data/feedback_data/model_metrics.json \
    MED_AI_MODEL_V2_FINETUNED_PATH=/app/outputs/checkpoints/eca_resnet50_v2_finetuned.pth \
    MED_AI_MODEL_V8_BEST_PATH=/app/outputs/checkpoints/eca_resnet50_v8_best_acc87_97.pth \
    MED_AI_DEVICE=auto

# Runtime libs commonly needed by Pillow/Matplotlib in Linux containers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 torchvision==0.17.2 && \
    pip install -r /app/requirements.txt

# Copy only runtime inference/UI files (exclude training scripts and dataset).
COPY app.py /app/app.py
COPY med_model_arch.py /app/med_model_arch.py
COPY med_clinical_ui.py /app/med_clinical_ui.py
COPY med_config.py /app/med_config.py
COPY med_feedback.py /app/med_feedback.py
COPY med_inference.py /app/med_inference.py
COPY med_metric_store.py /app/med_metric_store.py
COPY med_models.py /app/med_models.py
COPY med_report.py /app/med_report.py

# Model weights required for deployment inference.
RUN mkdir -p /app/outputs/checkpoints
COPY outputs/checkpoints/eca_resnet50_v2_finetuned.pth /app/outputs/checkpoints/eca_resnet50_v2_finetuned.pth
COPY outputs/checkpoints/eca_resnet50_v8_best_acc87_97.pth /app/outputs/checkpoints/eca_resnet50_v8_best_acc87_97.pth

# Persist clinical feedback outside container filesystem.
RUN mkdir -p /data/feedback_data

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
