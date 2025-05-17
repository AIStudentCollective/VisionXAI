import React from "react";
import { Input } from "@/components/ui/input"; // adjust import path if needed
import { Button } from "@/components/ui/button";

interface NameStepProps {
  fileName: string;
  setFileName: (name: string) => void;
  setStep: (step: number) => void;
}

const NameStep: React.FC<NameStepProps> = ({ fileName, setFileName, setStep }) => {
  return (
    <div className="flex justify-center items-center min-h-[60vh]">
      <div className="w-full max-w-2xl bg-[#210B2C] rounded-xl shadow-[0_0_6px_4px_rgba(141,57,235,0.5)] p-6 sm:p-8 md:p-10 flex flex-col gap-4">
        <h1 className="text-2xl font-light mb-1">File name</h1>
        <Input
          value={fileName}
          onChange={(e) => setFileName(e.target.value)}
          className="bg-[#210B2C] shadow-[0_0_5px_2px_rgba(255,255,255,0.3)] focus:outline-none"
        />
        <Button
          onClick={() => setStep(2)}
          className="self-center px-10 py-2 mt-4 font-light bg-gradient-to-r from-purple-600 to-indigo-500 rounded-lg shadow-md hover:opacity-90 transition"
        >
          Start
        </Button>
      </div>
    </div>
  );
};

export default NameStep;