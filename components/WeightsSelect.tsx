"use client"

import type React from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

interface WeightsSelectProps {
  weightsFile: File | null
  setWeightsFile: (file: File | null) => void
  weightsSelected: boolean
  setWeightsSelected: (selected: boolean) => void
  setStep: (step: number) => void
  steps: string[]
}

const WeightsSelect: React.FC<WeightsSelectProps> = ({
  weightsFile,
  setWeightsFile,
  weightsSelected,
  setWeightsSelected,
  setStep,
  steps,
}) => {
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setWeightsFile(event.target.files[0])
      setWeightsSelected(false) // Deselect default when uploading custom
    }
  }

  const handleDefaultSelect = () => {
    setWeightsSelected(true)
    setWeightsFile(null) // Clear file when selecting default
  }

  const handleCustomSelect = () => {
    setWeightsSelected(false)
  }

  return (
    <div className="flex flex-col h-full w-full">
      <h1 className="text-2xl font-light mb-6">Select Weights</h1>

      <div className="space-y-8">
        <div className="flex flex-col space-y-4">
          <p className="text-gray-300 text-base">
            Choose whether to use the default pre-trained weights or upload your own custom weights file.
          </p>

          <div className="flex flex-col md:flex-row gap-6 mt-4">
            {/* Default Weights Option */}
            <div
              className={`flex-1 p-6 border rounded-lg cursor-pointer transition-all ${
                weightsSelected
                  ? "border-purple-500 bg-[#2D1139]"
                  : "border-gray-700 bg-[#210B2C] hover:border-purple-400"
              }`}
              onClick={handleDefaultSelect}
            >
              <div className="flex items-start">
                <div>
                  <h3 className="text-white text-lg font-medium">Use Default Weights</h3>
                  <p className="text-gray-400 text-sm mt-1">
                    Use the standard pre-trained weights for the selected model architecture.
                  </p>
                </div>
              </div>
            </div>

            {/* Custom Weights Option */}
            <div
              className={`flex-1 p-6 border rounded-lg cursor-pointer transition-all ${
                !weightsSelected
                  ? "border-purple-500 bg-[#2D1139]"
                  : "border-gray-700 bg-[#210B2C] hover:border-purple-400"
              }`}
              onClick={handleCustomSelect}
            >
              <div className="flex items-start">
                <div>
                  <h3 className="text-white text-lg font-medium">Upload Custom Weights</h3>
                  <p className="text-gray-400 text-sm mt-1">
                    Upload your own pre-trained weights file (.pt, .pth, or .tar format).
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Custom Weights File Upload */}
        {!weightsSelected && (
          <div className="mt-6 p-6 border border-gray-700 rounded-lg bg-[#1A0922]">
            <Label htmlFor="weights_file" className="text-white text-base font-medium mb-3 block">
              Upload Weights File
            </Label>
            <Input
              id="weights_file"
              type="file"
              accept=".pt,.pth,.tar"
              className="block h-fit w-full text-sm text-gray-300 bg-[#210B2C] border border-purple-100 border-opacity-25 rounded-lg cursor-pointer file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-medium file:bg-purple-600 file:text-white hover:file:bg-purple-700"
              onChange={handleFileUpload}
            />
            {weightsFile ? (
              <p className="text-purple-400 text-sm mt-2">Selected: {weightsFile.name}</p>
            ) : (
              <p className="text-gray-400 text-sm mt-2">Supported formats: .pt, .pth, .tar</p>
            )}
          </div>
        )}
      </div>

      <div className="flex justify-between mt-auto pt-6">
        <Button
          className="bg-gradient-to-r from-purple-600 to-indigo-500 text-white text-base font-normal px-8 py-2 rounded-lg hover:opacity-90 transition"
          onClick={() => setStep(3)}
        >
          Back
        </Button>
        <Button
          className="bg-gradient-to-r from-purple-600 to-indigo-500 text-white text-base font-normal px-8 py-2 rounded-lg hover:opacity-90 transition"
          onClick={() => setStep(5)}
          disabled={!weightsSelected && !weightsFile}
        >
          Next
        </Button>
      </div>
    </div>
  )
}

export default WeightsSelect