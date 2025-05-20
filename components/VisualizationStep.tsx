"use client"

import type React from "react"
import { Button } from "@/components/ui/button"

interface VisualizationStepProps {
	visualizationUrl: string | null
	predictionInfo: string | null
	isLoading: boolean
	error: string | null
	handleSubmit: () => Promise<void>
	setStep: (step: number) => void
}

const VisualizationStep: React.FC<VisualizationStepProps> = ({
	visualizationUrl,
	predictionInfo,
	isLoading,
	error,
	handleSubmit,
	setStep,
}) => {
	return (
		<div className="flex flex-col h-full w-full">
		<h1 className="text-2xl font-light mb-6">Model Visualization</h1>
		
		<div className="flex flex-col items-center justify-center">
		{isLoading ? (
			<div className="text-center py-8">
			<div className="w-12 h-12 border-4 border-purple-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
			<p className="text-white">Generating visualization...</p>
			</div>
		) : error ? (
			<div className="text-center py-8">
			<p className="text-red-500 mb-4">{error}</p>
			<Button
			className="bg-gradient-to-r from-purple-600 to-indigo-500 text-white text-base font-normal px-8 py-2 rounded-lg hover:opacity-90 transition"
			onClick={handleSubmit}
			>
			Try Again
			</Button>
			</div>
		) : visualizationUrl ? (
			<div className="flex flex-col items-center">
			<div className="bg-[#2D1139] p-6 rounded-lg border border-purple-500 shadow-lg max-w-md">
			<img
			src={visualizationUrl || "/placeholder.svg"}
			alt="Model Visualization"
			className="max-w-full h-auto rounded-lg"
			/>
			{predictionInfo && (
				<div className="mt-4 p-3 bg-[#210B2C] rounded-md border border-purple-400">
				<p className="text-white text-center font-medium">{predictionInfo}</p>
				</div>
			)}
			</div>
			<div className="mt-4 text-gray-300 text-sm">
			<p>Manik -- put the LLM explanations here.</p>
			</div>
			</div>
		) : (
			<div className="text-center py-8">
			<p className="text-white mb-4">Click the button below to generate visualization</p>
			<Button
			className="bg-gradient-to-r from-purple-600 to-indigo-500 text-white text-base font-normal px-8 py-2 rounded-lg hover:opacity-90 transition"
			onClick={handleSubmit}
			>
			Generate Visualization
			</Button>
			</div>
		)}
		</div>
		
		<div className="flex justify-between mt-auto pt-6">
		<Button
		className="bg-gradient-to-r from-purple-600 to-indigo-500 text-white text-base font-normal px-8 py-2 rounded-lg hover:opacity-90 transition"
		onClick={() => setStep(4)}
		>
		Back
		</Button>
		{visualizationUrl && (
			<Button
			className="bg-gradient-to-r from-purple-600 to-indigo-500 text-white text-base font-normal px-8 py-2 rounded-lg hover:opacity-90 transition"
			onClick={() => {
				// Download functionality could be added here
				alert("Download functionality would be implemented here")
			}}
			>
			Download Results
			</Button>
		)}
		</div>
		</div>
	)
}

export default VisualizationStep