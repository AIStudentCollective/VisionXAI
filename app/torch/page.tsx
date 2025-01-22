import { SubmitButton } from "@/components/submit-button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export default async function Torch() {
  const data = await fetch("http://127.0.0.1:3000/api/pytorch/heatmap", {
    method: "GET",
    cache: "no-store",
  });

  let imageUrl: string | null = null;

  if (data.ok) {
    imageUrl = data.url; // The URL of the generated Grad-CAM image
  }

  return (
    <div className="flex items-center justify-center min-h-screen w-screen">
      <form
        className="flex-1 flex flex-col p-8 rounded w-full max-w-sm space-y-4"
        action="/api/pytorch/heatmap" 
        method="POST"
        encType="multipart/form-data"
      >
        <h1 className="text-2xl font-medium">Grad-CAM Visualization</h1>
        <p className="text-sm text-foreground">
          Input model details to generate Grad-CAM visualization.
        </p>

        <div className="flex flex-col gap-2 [&>input]:mb-3 mt-8">
          <Label htmlFor="model_name">Model Name</Label>
          <Input
            name="model_name"
            placeholder="e.g., densenet121"
            required
          />

          <Label htmlFor="weights_path">Weights File</Label>
          <Input
            type="file"
            name="weights_path"
            accept=".pth,.tar"
            required
          />

          <Label htmlFor="target_layer">Target Layer</Label>
          <Input
            name="target_layer"
            placeholder="e.g., features"
            required
          />

          <Label htmlFor="image_path">Image File</Label>
          <Input
            type="file"
            name="image_path"
            accept="image/*"
            required
          />

          <SubmitButton pendingText="Processing...">
            Generate
          </SubmitButton>
        </div>
      </form>

      {imageUrl && (
        <div className="mt-8">
          <h2 className="text-xl font-medium">Generated Heatmap</h2>
          <img
            src={imageUrl}
            alt="Grad-CAM Output"
            className="max-w-full max-h-[500px] border rounded shadow-lg"
          />
        </div>
      )}
    </div>
  );
}
