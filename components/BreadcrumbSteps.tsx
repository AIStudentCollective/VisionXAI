interface BreadcrumbStepsProps {
  steps: string[];
  currentStepIndex: number;
}

const BreadcrumbSteps: React.FC<BreadcrumbStepsProps> = ({
  steps,
  currentStepIndex,
}) => {
  return (
    <div className="flex items-center px-4 py-2 text-sm sm:text-base font-light text-gray-400">
      {steps.map((step, index) => (
        <div key={index} className="flex items-center">
          <span className={index === currentStepIndex ? "text-white font-normal" : ""}>
            {step}
          </span>
          {index < steps.length - 1 && (
            <span className="mx-2 text-gray-500">{">"}</span>
          )}
        </div>
      ))}
    </div>
  );
};

export default BreadcrumbSteps;
