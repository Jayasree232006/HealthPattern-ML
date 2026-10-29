import type { SymptomAssertion, SymptomUnderstandingResult, UnderstoodSymptom } from '../types';

const assertions: { value: SymptomAssertion; label: string }[] = [
  { value: 'affirmed', label: 'Affirmed' },
  { value: 'negated', label: 'Denied' },
  { value: 'uncertain', label: 'Uncertain' },
  { value: 'historical', label: 'Historical' },
];

const assertionLabels: Record<SymptomAssertion, string> = {
  affirmed: 'Affirmed',
  negated: 'Denied',
  uncertain: 'Uncertain',
  historical: 'Historical',
};

interface SymptomUnderstandingProps {
  understanding: SymptomUnderstandingResult;
  symptoms: UnderstoodSymptom[];
  onAssertionChange: (index: number, assertion: SymptomAssertion) => void;
  onRemove: (index: number) => void;
}

export function SymptomUnderstanding({
  understanding,
  symptoms,
  onAssertionChange,
  onRemove,
}: SymptomUnderstandingProps) {
  return (
    <section className="understanding" aria-labelledby="understanding-title">
      <div className="section-title-row">
        <div>
          <p className="step-label">02 / Review</p>
          <h2 id="understanding-title">What we understood</h2>
          <p className="section-intro">
            Check each interpretation. You can change its label or remove it from this review.
          </p>
        </div>
        <span className={`engine-label engine-${understanding.engine_used}`}>
          {understanding.engine_used === 'groq' ? 'Groq-assisted' : understanding.engine_used.replace(/_/g, ' ')}
        </span>
      </div>

      {understanding.clarification_needed && (
        <div className="clarification-callout" role="status">
          <strong>Clarification suggested</strong>
          <p>{understanding.clarification_prompt || 'The description may need more detail.'}</p>
        </div>
      )}

      {understanding.ambiguities.length > 0 && (
        <div className="ambiguity-list" aria-label="Ambiguous descriptions">
          {understanding.ambiguities.map((ambiguity, index) => (
            <p key={`${ambiguity}-${index}`}>{ambiguity}</p>
          ))}
        </div>
      )}

      {understanding.spelling_corrections.length > 0 && (
        <div className="correction-list" aria-label="Suggested spelling corrections">
          <h3>Possible wording corrections</h3>
          {understanding.spelling_corrections.map((correction, index) => (
            <p key={`${correction.original}-${index}`}>
              <span>{correction.original}</span>
              <span aria-hidden="true">→</span>
              <strong>{correction.suggested}</strong>
              {correction.clarification_recommended && <small>Check this suggestion</small>}
            </p>
          ))}
        </div>
      )}

      <div className="symptom-review-list">
        {symptoms.length === 0 ? (
          <p className="review-empty">No symptoms were extracted. Revise the description and try again.</p>
        ) : (
          symptoms.map((symptom, index) => (
            <article className="symptom-review-row" key={`${symptom.symptom_key}-${index}`}>
              <div className="symptom-review-copy">
                <span className={`assertion-dot assertion-${symptom.assertion}`} aria-hidden="true" />
                <div>
                  <strong>{symptom.display_name}</strong>
                  <span className="symptom-source">
                    {symptom.raw_mention || symptom.symptom_key.replace(/_/g, ' ')}
                    {symptom.temporal_context ? ` · ${symptom.temporal_context}` : ''}
                  </span>
                </div>
              </div>
              <div className="symptom-review-controls">
                <label className="sr-only" htmlFor={`assertion-${index}`}>
                  {symptom.display_name} assertion
                </label>
                <select
                  id={`assertion-${index}`}
                  value={symptom.assertion}
                  onChange={(event) => onAssertionChange(index, event.target.value as SymptomAssertion)}
                  aria-label={`${symptom.display_name} assertion, currently ${assertionLabels[symptom.assertion]}`}
                >
                  {assertions.map((assertion) => (
                    <option value={assertion.value} key={assertion.value}>
                      {assertion.label}
                    </option>
                  ))}
                </select>
                <button
                  className="remove-symptom"
                  type="button"
                  aria-label={`Remove ${symptom.display_name}`}
                  title={`Remove ${symptom.display_name} from this review`}
                  onClick={() => onRemove(index)}
                >
                  ×
                </button>
              </div>
            </article>
          ))
        )}
      </div>

      {understanding.notes && <p className="understanding-notes">{understanding.notes}</p>}
      {understanding.status !== 'success' && (
        <p className="understanding-status">Understanding status: {understanding.status.replace(/_/g, ' ')}</p>
      )}
    </section>
  );
}
