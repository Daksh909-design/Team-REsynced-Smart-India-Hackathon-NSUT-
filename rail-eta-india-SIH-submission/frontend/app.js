const $ = (id) => document.getElementById(id);
const form = $('search-form');
const trainInput = $('train-number');
const dateInput = $('journey-date');
const emptyState = $('empty-state');
const loadingState = $('loading-state');
const errorState = $('error-state');
const results = $('results');
const searchButton = $('search-button');
let lastQuery = null;

const todayIndia = () => {
  const now = new Date();
  const parts = new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Kolkata', year: 'numeric', month: '2-digit', day: '2-digit' }).formatToParts(now);
  return `${parts.find(p => p.type === 'year').value}-${parts.find(p => p.type === 'month').value}-${parts.find(p => p.type === 'day').value}`;
};
dateInput.value = todayIndia();

function show(view) {
  [emptyState, loadingState, errorState, results].forEach(node => node.classList.add('hidden'));
  view.classList.remove('hidden');
}
function formatTime(value) {
  return new Intl.DateTimeFormat('en-IN', { timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit', hour12: true }).format(new Date(value));
}
function formatDate(value) {
  return new Intl.DateTimeFormat('en-IN', { timeZone: 'Asia/Kolkata', day: '2-digit', month: 'short', year: 'numeric' }).format(new Date(`${value}T00:00:00+05:30`));
}
function safeNumber(value, fallback = '—') { return Number.isFinite(Number(value)) ? Number(value) : fallback; }

function render(data) {
  const finished = data.journey_status === 'finished';
  const notLive = data.journey_status === 'not_live';
  $('result-train').textContent = `Train ${data.train_number}`;
  $('result-date').textContent = formatDate(data.journey_date);
  $('current-station').textContent = data.current_station_code || 'In transit';
  $('current-delay').textContent = finished ? 'Finished' : notLive ? 'Not live' : safeNumber(data.current_delay_minutes, 0);
  $('current-delay-unit').textContent = finished || notLive ? '' : 'min';
  $('current-speed').textContent = finished || notLive ? (data.status_message || 'No active journey') : data.current_speed_kmh == null ? 'Live speed not published' : `${Math.round(data.current_speed_kmh)} km/h current speed`;
  const delay = Math.max(0, Number(data.current_delay_minutes) || 0);
  $('delay-track-fill').style.width = finished || notLive ? '0%' : `${Math.min(100, Math.max(8, delay / 2.4))}%`;
  $('model-name').textContent = finished || notLive ? 'No active forecast' : data.model_name.replaceAll('_', ' ');
  const highCount = data.etas.filter(stop => stop.confidence === 'high').length;
  const confidence = finished || notLive ? 'Not applicable' : highCount ? `High signal · ${highCount}/${data.etas.length}` : 'Standard signal';
  $('confidence-label').textContent = confidence;
  $('confidence-pill').classList.toggle('standard', finished || notLive || !highCount);
  $('model-note').textContent = data.status_message || (data.trained_through.includes('no timestamp') ? 'Cold-start prior + live delay propagation' : 'Timestamped live model');
  $('stop-count').textContent = data.etas.length;
  const age = data.source_freshness_seconds == null ? null : Math.round(data.source_freshness_seconds / 60);
  $('freshness-label').textContent = finished ? 'Journey finished' : notLive ? 'Journey not live' : age == null ? 'Live feed connected' : age <= 1 ? 'Updated just now' : `Source updated ${age} min ago`;
  if (!data.etas.length) {
    $('timeline').innerHTML = `<div class="finished-state">${data.status_message || 'There are no future halts to predict for this journey.'}</div>`;
    $('results-foot').classList.add('hidden');
    return;
  }
  $('results-foot').classList.remove('hidden');
  $('timeline').innerHTML = data.etas.slice(0, 12).map(stop => {
    const high = stop.confidence === 'high';
    const window = `${formatTime(stop.lower_arrival)} — ${formatTime(stop.upper_arrival)}`;
    return `<div class="timeline-row"><span class="timeline-marker ${high ? 'high' : ''}"></span><div class="timeline-station"><strong>${stop.station_code}</strong><span>STOP ${String(stop.sequence).padStart(2, '0')}${stop.remaining_distance_km == null ? '' : ` · ${Math.round(stop.remaining_distance_km)} KM`}</span></div><div class="eta-time"><strong>${formatTime(stop.predicted_arrival)}</strong><span>${stop.predicted_delay_minutes >= 0 ? '+' : ''}${stop.predicted_delay_minutes} min vs schedule</span></div><div class="eta-window"><span>RANGE</span><br />${window}</div><span class="mode-tag ${stop.prediction_mode}">${stop.prediction_mode.replace('_', ' ')}</span></div>`;
  }).join('');
  if (data.etas.length > 12) $('timeline').insertAdjacentHTML('beforeend', `<div class="timeline-more">+ ${data.etas.length - 12} more future stops · scroll the API response for the complete route</div>`);
}

async function fetchForecast() {
  const train = trainInput.value.trim();
  const date = dateInput.value;
  if (!/^\d{5}$/.test(train)) { $('error-title').textContent = 'Enter a 5-digit train number'; $('error-copy').textContent = 'For example: 12919'; show(errorState); return; }
  lastQuery = { train, date };
  show(loadingState); searchButton.disabled = true; searchButton.querySelector('span').textContent = 'Fetching…';
  try {
    const response = await fetch(`/v1/eta/${encodeURIComponent(train)}?journey_date=${encodeURIComponent(date)}`);
    const body = await response.text();
    let data;
    try { data = JSON.parse(body); } catch { data = { detail: body || 'The live provider did not return a forecast.' }; }
    if (!response.ok) throw new Error(data.detail || 'The live provider did not return a forecast.');
    render(data); show(results);
  } catch (error) {
    $('error-title').textContent = 'Couldn’t fetch this train'; $('error-copy').textContent = error.message || 'Try again in a moment.'; show(errorState);
  } finally { searchButton.disabled = false; searchButton.querySelector('span').textContent = 'Forecast ETA'; }
}
form.addEventListener('submit', (event) => { event.preventDefault(); fetchForecast(); });
$('retry-button').addEventListener('click', fetchForecast);
$('refresh-button').addEventListener('click', () => { if (lastQuery) fetchForecast(); });
