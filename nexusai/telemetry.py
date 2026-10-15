"""OpenTelemetry setup.

Traces and metrics go to an OTLP collector when OTEL_EXPORTER_OTLP_ENDPOINT is
set (for example Jaeger, Grafana Tempo or Honeycomb), to the console when
NEXUS_OTEL_CONSOLE=1, and nowhere otherwise. Instrumented code calls the
tracer and meter the same way in every case.
"""
from __future__ import annotations

import os

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

_configured = False


def setup_telemetry(service_name: str = "nexusai-gateway") -> None:
    global _configured
    if _configured:
        return
    _configured = True

    resource = Resource.create({"service.name": service_name})
    tracer_provider = TracerProvider(resource=resource)
    readers = []

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        readers.append(PeriodicExportingMetricReader(OTLPMetricExporter()))
    elif os.getenv("NEXUS_OTEL_CONSOLE") == "1":
        tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        readers.append(PeriodicExportingMetricReader(ConsoleMetricExporter(), export_interval_millis=30000))

    trace.set_tracer_provider(tracer_provider)
    metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=readers))


tracer = trace.get_tracer("nexusai")
meter = metrics.get_meter("nexusai")

requests_counter = meter.create_counter(
    "nexus.requests", description="Gateway requests by outcome, cache result and model"
)
provider_attempts = meter.create_counter(
    "nexus.provider.attempts", description="Upstream provider calls by provider and outcome"
)
tokens_counter = meter.create_counter("nexus.tokens", description="Tokens used, by model and direction")
cost_counter = meter.create_counter("nexus.cost_usd", unit="USD", description="Estimated spend by model")
latency_histogram = meter.create_histogram(
    "nexus.request.latency", unit="ms", description="End-to-end gateway latency"
)
