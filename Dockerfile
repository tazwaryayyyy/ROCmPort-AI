FROM node:20-bookworm AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
    COPY . .
COPY --from=frontend-build /app/frontend/dist ./frontend/dist
# Runtime envs: GROQ_API_KEY, ROCM_AVAILABLE, HIPCC_PATH, ROCPROF_PATH.
# Pass secrets at docker run/deploy time; do not bake .env into the image.
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
