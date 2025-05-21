"use client"

import React, { useState } from "react"
import MetadataCard from "@/components/MetadataCard"
import BreadcrumbSteps from "@/components/BreadcrumbSteps"
import NameStep from "@/components/NameStep"
import ImageUploadStep from "@/components/ImageUploadStep"
import ArchitectureSelectStep from "@/components/ArchitectureSelectStep"
import WeightsSelect from "@/components/WeightsSelect"
import VisualizationStep from "@/components/VisualizationStep"

const steps = ["Name", "Image Upload", "Architecture", "Weights", "Model Visualization"]

export default function TorchVIT() {
	const [step, setStep] = useState(1)
	const [fileName, setFileName] = useState("")
	const [modelName, setModelName] = useState("")
	const [weightsFile, setWeightsFile] = useState<File | null>(null)
	const [imageFile, setImageFile] = useState<File | null>(null)
	const [classLabelsFile, setClassLabelsFile] = useState<File | null>(null)
	const [visualizationUrl, setVisualizationUrl] = useState<string | null>(null)
	const [predictionInfo, setPredictionInfo] = useState<string | null>(null)
	const [isLoading, setIsLoading] = useState(false)
	const [error, setError] = useState<string | null>(null)
	const [architectureSelected, setArchitectureSelected] = useState(false)
	const [architectureType, setArchitectureType] = useState<"cnn" | "vit">("cnn")
	const [customModelFile, setCustomModelFile] = useState<File | null>(null)
	const [weightsSelected, setWeightsSelected] = useState(false)
	const [dragActive, setDragActive] = useState(false)
	const [search, setSearch] = React.useState("")
	const [open, setOpen] = useState(false)
	const [numClasses, setNumClasses] = useState<number>(0)
	const [imageSize, setImageSize] = useState<number>(0)
	
	const handleFileUpload =
	(
		setter: React.Dispatch<React.SetStateAction<File | null>>,
		setFlag?: React.Dispatch<React.SetStateAction<boolean>>,
	) =>
		(event: React.ChangeEvent<HTMLInputElement>) => {
		if (event.target.files) {
			setter(event.target.files[0])
			if (setFlag) setFlag(true)
			}
	}
	
	const handleSubmit = async () => {
		setIsLoading(true)
		setError(null)
		
		if ((!modelName && !customModelFile) || !imageFile || !classLabelsFile) {
			setError("Error: Model selection, image file, and class labels CSV are required.")
			setIsLoading(false)
			return
		}
		
		const formData = new FormData()

		if (customModelFile && weightsFile) {
			formData.append("custom_model_file", customModelFile)
			formData.append("custom_weights_file", weightsFile)
		} else if (modelName) {
			formData.append("model_name", modelName)
		}
		
		formData.append("image_path", imageFile as File)
		formData.append("class_labels_csv", classLabelsFile as File)
		
		// Add ViT specific parameters if needed
		if (numClasses > 0) {
			formData.append("num_classes", numClasses.toString())
		}
		
		if (imageSize > 0) {
			formData.append("image_size", imageSize.toString())
		}
		
		if (!customModelFile && !weightsSelected && weightsFile) {
			formData.append("weights_path", weightsFile)
		}
		
		try {
			let response;
			if (architectureType === "cnn") {
				response = await fetch("/api/pytorch/heatmap", { method: "POST", body: formData })
			}
			else
			{
				response = await fetch("/api/pytorch/heatmap_vit", { method: "POST", body: formData })
			}
			if (architectureType == 'cnn' && !response.ok) {
				throw new Error("Failed to generate Grad-CAM visualization.")
			}
			if (architectureType == 'vit' && !response.ok) {
				throw new Error("Failed to generate attention rollout visualization.")
			}
			const result = await response.json()
			setVisualizationUrl(`data:image/png;base64,${result.image}`)
			setPredictionInfo(`Predicted Class: ${result.predicted_class} (${result.predicted_probability})`)
			setStep(5) // Move to visualization step after successful API response

		} catch (error) {
			setError("Error processing request. Please try again.")
		} finally {
			setIsLoading(false)
		}
	}
	
	return (
		<div className="w-full px-4 pt-28 pb-8">
			{step === 1 && <NameStep fileName={fileName} setFileName={setFileName} setStep={setStep} />}
			
			{step === 2 && (
				<div className="flex flex-col md:flex-row gap-6 w-full max-w-6xl mx-auto">
					<div className="w-full md:w-1/3">
						<MetadataCard
							name={fileName}
							status="In Progress"
							id="41603"
							createdDate="5/22/2025"
							institution="UC Davis"
							creator="saparasa"
						/>
					</div>
					
					<div className="w-full md:w-2/3">
						<div className="bg-[#210B2C] rounded-xl shadow-[0_0_6px_4px_rgba(141,57,235,0.5)] p-6 sm:p-8 md:p-10 flex flex-col">
							<BreadcrumbSteps steps={steps} currentStepIndex={step - 1} />
							<div className="mt-6">
								<ImageUploadStep imageFile={imageFile} setImageFile={setImageFile} setStep={setStep} />
							</div>
						</div>
					</div>
				</div>
			)}
			
			{step === 3 && (
				<div className="flex flex-col md:flex-row gap-6 w-full max-w-6xl mx-auto">
					<div className="w-full md:w-1/3">
						<MetadataCard
							name={fileName}
							status="In Progress"
							id="41603"
							createdDate="5/22/2025"
							institution="UC Davis"
							creator="saparasa"
						/>
					</div>
				
					<div className="w-full md:w-2/3">
						<div className="bg-[#210B2C] rounded-xl shadow-[0_0_6px_4px_rgba(141,57,235,0.5)] p-6 sm:p-8 md:p-10 flex flex-col">
							<BreadcrumbSteps steps={steps} currentStepIndex={step - 1} />
							<div className="mt-6">
								<ArchitectureSelectStep
									modelName={modelName}
									setModelName={setModelName}
									weightsFile={weightsFile}
									setWeightsFile={setWeightsFile}
									classLabelsFile={classLabelsFile}
									setClassLabelsFile={setClassLabelsFile}
									customModelFile={customModelFile}
									setCustomModelFile={setCustomModelFile}
									setStep={setStep}
									numClasses={numClasses}
									setNumClasses={setNumClasses}
									imageSize={imageSize}
									setImageSize={setImageSize}
									architectureType={architectureType}
									setArchitectureType={setArchitectureType}
								/>
							</div>
						</div>
					</div>
				</div>
			)}
			
			{step === 4 && (
				<div className="flex flex-col md:flex-row gap-6 w-full max-w-6xl mx-auto">
					<div className="w-full md:w-1/3">
						<MetadataCard
							name={fileName}
							status="In Progress"
							id="41603"
							createdDate="5/22/2025"
							institution="UC Davis"
							creator="saparasa"
						/>
					</div>
					
					<div className="w-full md:w-2/3">
						<div className="bg-[#210B2C] rounded-xl shadow-[0_0_6px_4px_rgba(141,57,235,0.5)] p-6 sm:p-8 md:p-10 flex flex-col">
							<BreadcrumbSteps steps={steps} currentStepIndex={step - 1} />
							<div className="mt-6">
								<WeightsSelect
									weightsFile={weightsFile}
									setWeightsFile={setWeightsFile}
									weightsSelected={weightsSelected}
									setWeightsSelected={setWeightsSelected}
									setStep={setStep}
									steps={steps}
									isLoading={isLoading}
									handleSubmit={handleSubmit}
								/>
							</div>
						</div>
					</div>
				</div>
			)}
			
			{step === 4.5 && (
				<div className="flex flex-col md:flex-row gap-6 w-full max-w-6xl mx-auto">
					<div className="w-full md:w-1/3">
						<MetadataCard
							name={fileName}
							status="Processing"
							id="41603"
							createdDate="5/22/2025"
							institution="UC Davis"
							creator="saparasa"
						/>
					</div>
					
					<div className="w-full md:w-2/3">
						<div className="bg-[#210B2C] rounded-xl shadow-[0_0_6px_4px_rgba(141,57,235,0.5)] p-6 sm:p-8 md:p-10 flex flex-col">
							<BreadcrumbSteps steps={steps} currentStepIndex={3.5} />
							<div className="mt-6 flex flex-col items-center justify-center py-12">
								<div className="w-16 h-16 border-4 border-purple-500 border-t-transparent rounded-full animate-spin mb-6"></div>
								<p className="text-white text-xl">Processing your request...</p>
								<p className="text-gray-400 mt-2">This may take a moment</p>
							</div>
						</div>
					</div>
				</div>
			)}
			
			{step === 5 && (
				<div className="flex flex-col md:flex-row gap-6 w-full max-w-6xl mx-auto">
					<div className="w-full md:w-1/3">
						<MetadataCard
							name={fileName}
							status={visualizationUrl ? "Completed" : "In Progress"}
							id="41603"
							createdDate="5/22/2025"
							institution="UC Davis"
							creator="saparasa"
						/>
					</div>
					
					<div className="w-full md:w-2/3">
						<div className="bg-[#210B2C] rounded-xl shadow-[0_0_6px_4px_rgba(141,57,235,0.5)] p-6 sm:p-8 md:p-10 flex flex-col">
							<BreadcrumbSteps steps={steps} currentStepIndex={step - 1} />
							<div className="mt-6">
								<VisualizationStep
									visualizationUrl={visualizationUrl}
									predictionInfo={predictionInfo}
									isLoading={isLoading}
									error={error}
									handleSubmit={handleSubmit}
									setStep={setStep}
								/>
							</div>
						</div>
					</div>
				</div>
			)}
		</div>
	)
}
