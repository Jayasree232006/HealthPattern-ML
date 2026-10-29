import { useState } from 'react';
import './styles/global.css';
import { predictConditions } from './api/prediction';
import type {
  AgeGroup,
  DurationUnit,
  PredictRequest,
  PredictResponse,
  SymptomAssertion,
  UnderstoodSymptom,
} from './types';
import {
  CommonPredictions,
  EducationalDisclaimer,
  RareConditionMatches,
  SymptomUnderstanding,
} from './components';

type Stage = 'describe' | 'review' | 'results';

const ageGroups: { label: string; value: AgeGroup | '' }[] = [
  { label: 'Prefer not to say', value: '' },
  { label: 'Infant', value: 'infant' },
  { label: 'Child', value: 'child' },
  { label: 'Adolescent', value: 'adolescent' },
  { label: 'Adult', value: 'adult' },
  { label: 'Senior', value: 'senior' },
];

const durationOptions: { label: string; value: number; unit: DurationUnit }[] = [
  { label: 'Under one day', value: 12, unit: 'hours' },
  { label: '1–3 days', value: 2, unit: 'days' },
  { label: '4–7 days', value: 5, unit: 'days' },
  { label: '1–2 weeks', value: 1, unit: 'weeks' },
  { label: 'More than 2 weeks', value: 3, unit: 'weeks' },
  { label: 'Several months', value: 3, unit: 'months' },
];

const assertionText: Record<SymptomAssertion, string> = {
  affirmed: 'affirmed',
  negated: 'denied',
  uncertain: 'uncertain',
  historical: 'historical',
};

function App() {
  const [description, setDescription] = useState('');
  const [ageGroup, setAgeGroup] = useState<AgeGroup | ''>('');
  const [durationIndex, setDurationIndex] = useState('');
  const [stage, setStage] = useState<Stage>('describe');
  const [response, setResponse] = useState<PredictResponse | null>(null);
  const [predictionResponse, setPredictionResponse] = useState<PredictResponse | null>(null);
  const [reviewedSymptoms, setReviewedSymptoms] = useState<UnderstoodSymptom[]>([]);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError('');
    setResponse(null);
    setPredictionResponse(null);
    setReviewedSymptoms([]);

    const freeText = description.trim();
    if (!freeText) {
      setError('Describe at least one symptom to continue.');
      return;
    }

    const selectedDuration = durationOptions[Number(durationIndex)];
    const request: PredictRequest = {
      free_text_symptoms: freeText,
      age_group: ageGroup || null,
      duration: selectedDuration
        ? { value: selectedDuration.value, unit: selectedDuration.unit }
        : null,
    };

    setIsLoading(true);
    try {
      const data = await predictConditions(request);
      if (data.status === 'error') {
        throw new Error(data.message || 'The prediction service could not process this request.');
      }
      if (!data.symptom_understanding) {
        throw new Error('The service returned no symptom-understanding result. No predictions are being shown.');
      }
      setResponse(data);
      setReviewedSymptoms(data.symptom_understanding.symptoms);
      setStage('review');
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : 'Unable to communicate with the prediction service. Check the connection and try again.',
      );
      setStage('describe');
    } finally {
      setIsLoading(false);
    }
  };

  const handleConfirmSymptoms = async () => {
    const confirmedSymptoms = [...new Set(
      reviewedSymptoms
        .filter((symptom) => symptom.assertion === 'affirmed')
        .map((symptom) => symptom.symptom_key),
    )];

    if (confirmedSymptoms.length === 0) {
      setError('Mark at least one symptom as affirmed to request pattern predictions.');
      return;
    }

    setError('');
    setIsLoading(true);
    const selectedDuration = durationOptions[Number(durationIndex)];
    const request: PredictRequest = {
      free_text_symptoms: null,
      structured_symptoms: confirmedSymptoms,
      age_group: ageGroup || null,
      duration: selectedDuration
        ? { value: selectedDuration.value, unit: selectedDuration.unit }
        : null,
    };

    try {
      const data = await predictConditions(request);
      if (data.status === 'error') {
        throw new Error(data.message || 'The prediction service could not process the confirmed symptoms.');
      }
      setPredictionResponse(data);
      setStage('results');
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : 'Unable to send the confirmed symptoms to the prediction service.',
      );
    } finally {
      setIsLoading(false);
    }
  };

  const updateAssertion = (index: number, assertion: SymptomAssertion) => {
    setReviewedSymptoms((current) =>
      current.map((symptom, itemIndex) =>
        itemIndex === index ? { ...symptom, assertion } : symptom,
      ),
    );
  };

  const removeSymptom = (index: number) => {
    setReviewedSymptoms((current) => current.filter((_, itemIndex) => itemIndex !== index));
  };

  const resetToDescription = () => {
    setStage('describe');
    setError('');
  };

  const understanding = response?.symptom_understanding;

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#home" aria-label="Rare & Common Disease Detector home">
          <span className="brand-mark" aria-hidden="true">+</span>
          <span className="brand-full">Rare &amp; Common Disease Detector<span className="brand-period">.</span></span>
          <span className="brand-short">Disease Pattern Detector<span className="brand-period">.</span></span>
        </a>
        <div className="prototype-label">
          <span className="live-dot" aria-hidden="true" />
          Live educational service
        </div>
      </header>

      <main id="home" className="workspace">
        <section className="intro-row" aria-labelledby="page-title">
          <div>
            <p className="eyebrow">Experimental educational prototype</p>
            <h1 id="page-title">Disease Pattern Detector</h1>
          </div>
          <p className="intro-copy">
            Describe symptoms in your own words. AI-assisted symptom understanding helps you review what was extracted, then explore experimental ML pattern scores for five common conditions and separate literature-based matches for four selected rare conditions.
          </p>
        </section>

        <nav className="stepper" aria-label="Exploration steps">
          {(['describe', 'review', 'results'] as Stage[]).map((item, index) => (
            <div className={`stepper-item${stage === item ? ' is-current' : ''}${(['review', 'results'] as Stage[]).indexOf(stage) > index ? ' is-complete' : ''}`} key={item}>
              <span>{`0${index + 1}`}</span>
              {item === 'describe' ? 'Describe' : item === 'review' ? 'Review' : 'Patterns'}
            </div>
          ))}
        </nav>

        {stage === 'describe' && (
          <div className="intake-layout">
            <section className="intake-panel" aria-labelledby="intake-title">
              <div className="section-title-row">
                <div>
                  <p className="step-label">01 / Describe</p>
                  <h2 id="intake-title">What have you noticed?</h2>
                </div>
                <span className="privacy-mark">Not saved</span>
              </div>

              <form onSubmit={handleSubmit} noValidate>
                <label className="field-label" htmlFor="symptom-description">
                  Your description <span>Required</span>
                </label>
                <textarea
                  id="symptom-description"
                  value={description}
                  onChange={(event) => setDescription(event.target.value)}
                  placeholder="For example: I've had a scratchy throat and a mild headache since yesterday."
                  rows={6}
                  maxLength={1200}
                  aria-describedby="description-hint"
                  required
                />
                <div className="input-meta" id="description-hint">
                  <span>Include what is happening and when it started, if known.</span>
                  <span>{description.length}/1200</span>
                </div>

                <div className="detail-fields">
                  <div className="field-group">
                    <label className="field-label" htmlFor="age-group">Age group <span>Optional</span></label>
                    <select id="age-group" value={ageGroup} onChange={(event) => setAgeGroup(event.target.value as AgeGroup | '')}>
                      {ageGroups.map((option) => <option key={option.label} value={option.value}>{option.label}</option>)}
                    </select>
                  </div>
                  <div className="field-group">
                    <label className="field-label" htmlFor="duration">Duration <span>Optional</span></label>
                    <select id="duration" value={durationIndex} onChange={(event) => setDurationIndex(event.target.value)}>
                      <option value="">Choose duration</option>
                      {durationOptions.map((option, index) => <option key={option.label} value={index}>{option.label}</option>)}
                    </select>
                  </div>
                </div>

                {error && <div className="form-error" role="alert">{error}</div>}

                <button className="primary-button" type="submit" disabled={isLoading}>
                  {isLoading ? <><span className="loader-ring" aria-hidden="true" />Understanding symptoms…</> : <>Understand my description <span aria-hidden="true">→</span></>}
                </button>
              </form>
            </section>

            <aside className="side-note">
              <span className="side-note-mark" aria-hidden="true">i</span>
              <div>
                <h2>Understand, then confirm</h2>
                <p>The first request returns an interpretation and an initial model response. We hold those patterns while you review the interpretation.</p>
                <p className="small-note">Confirmation sends only affirmed symptom tags as structured input for a fresh prediction request. It does not rerun Groq on your edits.</p>
              </div>
            </aside>
          </div>
        )}

        {stage === 'review' && response && understanding && (
          <div className="review-layout">
            <div className="review-main">
              <SymptomUnderstanding
                understanding={understanding}
                symptoms={reviewedSymptoms}
                onAssertionChange={updateAssertion}
                onRemove={removeSymptom}
              />
              <div className="review-actions">
                <button className="text-button" type="button" onClick={resetToDescription}>← Revise description</button>
                {error && <div className="form-error" role="alert">{error}</div>}
                <button className="primary-button" type="button" onClick={handleConfirmSymptoms} disabled={isLoading}>
                  {isLoading ? <><span className="loader-ring" aria-hidden="true" />Generating confirmed patterns…</> : <>Confirm symptoms &amp; request patterns <span aria-hidden="true">→</span></>}
                </button>
              </div>
            </div>
            <aside className="side-note review-notice">
              <span className="side-note-mark" aria-hidden="true">!</span>
              <div>
                <h2>About your edits</h2>
                <p>Confirmation sends the current affirmed symptom tags as structured input in a new request. Removed, denied, uncertain, and historical items are left out.</p>
                {understanding.clarification_needed && (
                  <p className="small-note">A clarification was requested. Revise your description and submit it again if you want the service to process that additional context.</p>
                )}
              </div>
            </aside>
          </div>
        )}

        {stage === 'results' && response && understanding && predictionResponse && (
          <section className="results-view" aria-labelledby="results-title" aria-live="polite">
            <div className="results-view-header">
              <div>
                <p className="step-label">03 / Patterns</p>
                <h2 id="results-title">Returned exploration</h2>
              </div>
              <button className="text-button" type="button" onClick={resetToDescription}>Start over</button>
            </div>

            <div className="response-notice" role="note">
              These patterns were requested after confirmation using only the reviewed symptoms labeled affirmed. The backend received those canonical tags as structured input; it did not re-run Groq on your edits or reinterpret the original narrative.
            </div>

            {predictionResponse.status === 'model_unavailable' ? (
              <div className="service-state" role="status">
                <strong>Prediction model unavailable</strong>
                <p>{predictionResponse.message}</p>
                <p>Symptom understanding is shown above; no condition predictions are available for this response.</p>
              </div>
            ) : (
              <>
                <EducationalDisclaimer disclaimerText={predictionResponse.disclaimer} />
                <CommonPredictions predictions={predictionResponse.predictions} modelVersion={predictionResponse.model_version} />
                <RareConditionMatches matches={predictionResponse.rare_condition_matches} />
              </>
            )}

            <section className="reviewed-summary" aria-label="Reviewed symptom summary">
              <h3>Your reviewed interpretation</h3>
              <div className="reviewed-chips">
                {reviewedSymptoms.length ? reviewedSymptoms.map((symptom, index) => (
                  <span className={`reviewed-chip assertion-${symptom.assertion}`} key={`${symptom.symptom_key}-${index}`}>
                    <strong>{assertionText[symptom.assertion]}</strong> {symptom.display_name}
                  </span>
                )) : <span className="small-note">No symptoms retained in your review.</span>}
              </div>
            </section>
            <p className="response-meta">Understanding engine: <strong>{understanding.engine_used.replace(/_/g, ' ')}</strong> · Prediction request: <strong>{predictionResponse.status}</strong></p>
          </section>
        )}

        <footer className="page-disclaimer">
          <strong>Educational use only.</strong> Rare &amp; Common Disease Detector is an experimental prototype, not a medical diagnostic tool. Common-condition model scores are experimental and are not clinical probabilities; rare-condition matches are literature-based indicators, not diagnoses. Consult a qualified healthcare professional about health concerns.
        </footer>
      </main>
    </div>
  );
}

export default App;