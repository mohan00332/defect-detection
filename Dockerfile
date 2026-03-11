FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
  PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1

RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates libgl1 libglib2.0-0 \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 3000

CMD ["sh", "-c", "gunicorn -w 1 -b 0.0.0.0:${PORT:-3000} --timeout 180 app:app"]
