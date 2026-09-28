import { createFileRoute } from "@tanstack/react-router";
import { CloudRain, CloudSun, Gauge, Sparkles, Wind } from "lucide-react";
import { FormEvent, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";

type Model = "Random Forest" | "Decision Tree" | "XGBoost";
type WeatherInputs = {
  Temperature: number;
  Humidity: number;
  Wind_Speed: number;
  Cloud_Cover: number;
  Pressure: number;
};
type Prediction = {
  model: Model;
  prediction: string;
  prediction_code: number;
  rain_probability: number;
};

const API_URL = import.meta.env["VITE_WEATHER_API_URL"] ?? "http://127.0.0.1:8000";

const defaultInputs: WeatherInputs = {
  Temperature: 23.22237,
  Humidity: 76.87794,
  Wind_Speed: 15.82567,
  Cloud_Cover: 72.86979,
  Pressure: 980.1089,
};

const modelDetails = [
  { name: "Random Forest" as Model, short: "RF", note: "Balanced", description: "A robust ensemble that stays steady across noisy atmospheric readings." },
  { name: "Decision Tree" as Model, short: "DT", note: "Direct", description: "Clear branching logic for a fast, easy-to-interpret prediction." },
  { name: "XGBoost" as Model, short: "XG", note: "Precise", description: "Gradient-boosted precision for subtle, fast-changing weather patterns." },
];

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Aetheria — Intelligent Weather Prediction Engine" },
      { name: "description", content: "Predict atmospheric rainfall conditions in real-time with machine-learning models (Random Forest, Decision Tree, XGBoost)." },
      { property: "og:title", content: "Aetheria — AI Weather Prediction" },
      { property: "og:description", content: "Turn live atmospheric readings into instant ML-driven weather forecasts." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

function Index() {
  const [inputs, setInputs] = useState(defaultInputs);
  const [model, setModel] = useState<Model>("Random Forest");
  const [result, setResult] = useState<Prediction>({ model: "Random Forest", prediction: "rain", prediction_code: 1, rain_probability: 0.87 });
  const [status, setStatus] = useState<"checking" | "online" | "offline">("checking");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const predictorRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), 3000);
    fetch(`${API_URL}/health`, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error("Health check failed");
        setStatus("online");
      })
      .catch(() => setStatus("offline"))
      .finally(() => window.clearTimeout(timer));
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, []);

  const setValue = (key: keyof WeatherInputs, value: string) => {
    setInputs((current) => ({ ...current, [key]: Number(value) }));
  };

  async function predict(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsLoading(true);
    setError("");
    try {
      const response = await fetch(`${API_URL}/predict/model`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...inputs, model }),
      });
      if (!response.ok) throw new Error("Prediction service returned an error");
      const data = (await response.json()) as Prediction;
      setResult(data);
      setStatus("online");
    } catch {
      setStatus("offline");
      setError("The prediction API is unavailable. Start the backend on port 8000, then try again.");
    } finally {
      setIsLoading(false);
    }
  }

  const probability = Math.round(result.rain_probability * 100);
  const isRain = result.prediction.toLowerCase().includes("rain");

  return (
    <main className="weather-shell relative min-h-screen overflow-hidden bg-background text-foreground">
      <div className="weather-atmosphere" aria-hidden="true">
        <span className="weather-aura weather-aura-one" />
        <span className="weather-aura weather-aura-two" />
        <span className="weather-aura weather-aura-three" />
        <span className="weather-cloud weather-cloud-one" />
        <span className="weather-cloud weather-cloud-two" />
      </div>

      <div className="relative z-10 mx-auto max-w-6xl px-5 sm:px-8">
        <header className="flex items-center justify-between py-6">
          <a href="#top" className="flex items-center gap-3" aria-label="Aetheria home">
            <span className="glass grid size-11 place-items-center rounded-2xl text-primary"><CloudSun className="size-5" /></span>
            <span className="leading-tight">
              <span className="block font-display text-lg font-semibold">Aetheria</span>
              <span className="hidden text-[11px] uppercase text-muted-foreground sm:block">Prediction engine</span>
            </span>
          </a>
          <div className="flex items-center gap-3">
            <div className={`status-pill status-${status}`} aria-label={`API status: ${status}`}>
              <span className="status-dot" />{status === "checking" ? "Checking API" : `API ${status}`}
            </div>
            <Button variant="frost" size="lg" onClick={() => predictorRef.current?.scrollIntoView({ behavior: "smooth" })}><span className="hidden sm:inline">Start predicting</span><span className="sm:hidden">Predict</span></Button>
          </div>
        </header>

        <section id="top" className="grid items-center gap-12 py-10 lg:grid-cols-12 lg:py-16">
          <div className="animate-rise lg:col-span-6">
            <div className="glass inline-flex items-center gap-2 rounded-full px-4 py-2 text-xs text-muted-foreground">
              <Sparkles className="size-3.5 text-accent" /> Three machine-learning models, one clear forecast
            </div>
            <h1 className="mt-6 max-w-xl font-display text-5xl font-bold leading-[1.05] sm:text-6xl lg:text-7xl">
              Read the sky <span className="block text-accent">before it changes.</span>
            </h1>
            <p className="mt-6 max-w-lg text-base leading-relaxed text-muted-foreground sm:text-lg">
              Turn five atmospheric readings into an immediate rain forecast, powered by your choice of prediction model.
            </p>
            <div className="mt-8 flex flex-wrap gap-4">
              <Button variant="weather" size="xl" onClick={() => predictorRef.current?.scrollIntoView({ behavior: "smooth" })}>Make a prediction <CloudRain /></Button>
              <Button variant="glass" size="xl" onClick={() => window.open(`${API_URL}/health`, "_blank", "noopener,noreferrer")}>View API health <Gauge /></Button>
            </div>
            <div className="mt-10 grid max-w-lg grid-cols-3 gap-3">
              <Metric value="5" label="Weather inputs" />
              <Metric value="3" label="ML models" />
              <Metric value="Live" label="API status" />
            </div>
          </div>

          <section ref={predictorRef} className="glass animate-rise-delay relative rounded-[28px] p-5 sm:p-8 lg:col-span-6" aria-labelledby="predict-title">
            <div className="flex items-start justify-between gap-4">
              <div><p className="text-xs uppercase text-muted-foreground">Live prediction</p><h2 id="predict-title" className="mt-1 font-display text-xl font-semibold">Atmospheric model</h2></div>
              <span className="model-pill">{result.model}</span>
            </div>

            <form onSubmit={predict} className="mt-6">
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                <WeatherField label="Temperature" unit="°C" value={inputs.Temperature} min={-60} max={70} onChange={(value) => setValue("Temperature", value)} />
                <WeatherField label="Humidity" unit="%" value={inputs.Humidity} min={0} max={100} onChange={(value) => setValue("Humidity", value)} />
                <WeatherField label="Wind speed" unit="km/h" value={inputs.Wind_Speed} min={0} max={400} onChange={(value) => setValue("Wind_Speed", value)} />
                <WeatherField label="Cloud cover" unit="%" value={inputs.Cloud_Cover} min={0} max={100} onChange={(value) => setValue("Cloud_Cover", value)} />
                <WeatherField label="Pressure" unit="hPa" value={inputs.Pressure} min={850} max={1100} onChange={(value) => setValue("Pressure", value)} />
                <label className="weather-field"><span>Model</span><select value={model} onChange={(event) => setModel(event.target.value as Model)}>{modelDetails.map((item) => <option key={item.name}>{item.name}</option>)}</select></label>
              </div>
              <Button variant="weather" size="xl" className="mt-4 w-full" disabled={isLoading} type="submit">{isLoading ? "Reading the sky…" : "Predict weather"}<Wind /></Button>
            </form>

            {error ? <p className="mt-4 rounded-xl border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive-foreground" role="alert">{error}</p> : null}

            <div className="result-panel mt-5" aria-live="polite">
              <div className="forecast-orbit">
                <span className="forecast-ring ring-one" /><span className="forecast-ring ring-two" /><span className="forecast-ring ring-three" />
                {isRain ? <CloudRain className="size-10 text-accent" /> : <CloudSun className="size-10 text-sun" />}
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-end justify-between gap-4"><div><p className="text-xs text-muted-foreground">Rain probability</p><p className="font-display text-4xl font-bold text-accent">{probability}%</p></div><span className="code-pill">code: {result.prediction_code}</span></div>
                <progress className="probability-bar mt-4" max="100" value={probability} aria-label={`${probability}% rain probability`} />
                <div className="mt-4 flex items-center justify-between text-sm"><span className="capitalize text-muted-foreground">Forecast</span><strong className="capitalize">{result.prediction}</strong></div>
              </div>
            </div>
          </section>
        </section>

        <section className="py-10" aria-labelledby="models-title">
          <div className="mb-6 flex items-end justify-between gap-4"><div><h2 id="models-title" className="font-display text-2xl font-semibold">Choose your model</h2><p className="mt-1 text-sm text-muted-foreground">Pick the prediction approach that fits your reading.</p></div><span className="hidden text-xs uppercase text-muted-foreground sm:block">POST /predict/model</span></div>
          <div className="grid gap-5 md:grid-cols-3">
            {modelDetails.map((item) => <button key={item.name} type="button" className={`model-card glass ${model === item.name ? "model-card-selected" : ""}`} onClick={() => { setModel(item.name); predictorRef.current?.scrollIntoView({ behavior: "smooth" }); }}><span className="flex items-center justify-between"><span className="model-mark">{item.short}</span><span className="code-pill">{item.note}</span></span><strong className="mt-5 block font-display text-lg">{item.name}</strong><span className="mt-2 block text-sm leading-relaxed text-muted-foreground">{item.description}</span><span className="mt-5 flex items-center gap-2 text-sm text-accent">Use this model <span aria-hidden="true">→</span></span></button>)}
          </div>
        </section>

        <footer className="flex flex-col items-center justify-between gap-3 border-t border-border py-8 text-sm text-muted-foreground sm:flex-row">
          <p className="font-display text-foreground">Aetheria Weather AI</p><p className="font-mono text-xs">GET /health · POST /predict/model</p><p>Atmospheric intelligence, made visible.</p>
        </footer>
      </div>
    </main>
  );
}

function Metric({ value, label }: { value: string; label: string }) {
  return <div className="glass rounded-2xl px-4 py-4"><p className="font-display text-2xl font-bold">{value}</p><p className="mt-1 text-[11px] uppercase text-muted-foreground">{label}</p></div>;
}

function WeatherField({ label, unit, value, min, max, onChange }: { label: string; unit: string; value: number; min: number; max: number; onChange: (value: string) => void }) {
  return <label className="weather-field"><span>{label} <em>{unit}</em></span><input required type="number" step="any" value={value} min={min} max={max} onChange={(event) => onChange(event.target.value)} /></label>;
}