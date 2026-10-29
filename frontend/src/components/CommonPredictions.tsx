import React from 'react';
import type { ConditionPrediction } from '../types';

interface CommonPredictionsProps {
  predictions: ConditionPrediction[];
  modelVersion?: string | null;
}

export const CommonPredictions: React.FC<CommonPredictionsProps> = ({
  predictions,
  modelVersion,
}) => {
  if (predictions.length === 0) {
    return (
      <div className="empty-substate">
        <p>No common-condition pattern matches identified for the provided symptoms.</p>
      </div>
    );
  }

  return (
    <div className="predictions-group">
      <div className="group-header">
        <div>
          <h3 className="group-title">Common Condition Patterns</h3>
          <p className="group-subtitle">
            Experimental model indicators based on shared common symptom profiles.
          </p>
        </div>
        {modelVersion && <span className="version-pill">v{modelVersion}</span>}
      </div>

      <div className="cards-stack">
        {predictions.map((pred, index) => {
          const scorePercent = Math.round(pred.confidence * 100);

          return (
            <article className="candidate-card" key={`${pred.condition}-${index}`}>
              <div className="candidate-topline">
                <span className="candidate-index">PATTERN 0{index + 1}</span>
                <span
                  className="score-pill common-score"
                  title="Experimental machine learning model ranking score. Not a diagnostic likelihood or clinical probability."
                >
                  <span className="score-val">{scorePercent}%</span>
                  <span className="score-desc">experimental model score</span>
                </span>
              </div>

              <h4 className="candidate-name">{pred.condition}</h4>

              {pred.description && (
                <p className="candidate-summary">{pred.description}</p>
              )}

              <div className="score-bar-track" aria-hidden="true">
                <div
                  className="score-bar-fill common-bar"
                  style={{ width: `${Math.max(4, scorePercent)}%` }}
                />
              </div>

              <div className="candidate-footer-note">
                <span>Indicative pattern score only • Not a clinical diagnosis</span>
              </div>
            </article>
          );
        })}
      </div>
    </div>
  );
};
