import React from 'react';

interface EducationalDisclaimerProps {
  disclaimerText?: string;
}

export const EducationalDisclaimer: React.FC<EducationalDisclaimerProps> = ({
  disclaimerText,
}) => {
  return (
    <div className="educational-banner" role="region" aria-label="Educational Disclaimer">
      <div className="banner-icon-col" aria-hidden="true">
        <span className="banner-icon">ℹ</span>
      </div>
      <div className="banner-body">
        <h4 className="banner-title">Educational Prototype Notice</h4>
        <p className="banner-text">
          {disclaimerText ||
            'This prototype is strictly for educational and research purposes. It does not provide medical diagnosis, clinical likelihoods, or personalized medical advice.'}
        </p>
        <p className="banner-subtext">
          Predicted scores and literature matches reflect automated computational heuristics and experimental models, not clinical probabilities. Always consult a qualified medical provider for personal health concerns or diagnosis.
        </p>
      </div>
    </div>
  );
};
