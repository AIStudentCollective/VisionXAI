import { signInAction } from "@/app/actions"
import { FormMessage, type Message } from "@/components/form-message"
import { SubmitButton } from "@/components/submit-button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import Link from "next/link"

export default async function Login(props: { searchParams: Promise<Message> }) {
  const searchParams = await props.searchParams

  if ("message" in searchParams) {
    return (
      <div className="w-full flex-1 flex items-center h-screen sm:max-w-md mx-auto justify-center gap-2 p-4">
        <FormMessage message={searchParams} />
      </div>
    )
  }

  return (
    <div className="flex flex-col w-full h-screen sm:max-w-md mx-auto gap-2 p-4 min-h-screen">
      <div className="flex-1 flex items-center justify-center px-4">
        <form className="w-full max-w-md p-8 rounded bg-white shadow-md space-y-4">
          <h1 className="text-2xl text-[#9333EA] font-medium">Sign in</h1>
          <div className="flex flex-col gap-2 [&>input]:mb-3 mt-8 text-black">
            <Label htmlFor="email">Email</Label>
            <Input name="email" placeholder="you@example.com" required />
            <div className="flex justify-between items-center">
              <Label htmlFor="password">Password</Label>
              <Link className="text-xs text-foreground underline" href="/forgot-password">
                Forgot Password?
              </Link>
            </div>
            <Input type="password" name="password" placeholder="Your password" required />
            <SubmitButton
              className="bg-gradient-to-r from-purple-600 to-indigo-500"
              pendingText="Signing in..."
              formAction={signInAction}
            >
              Sign in
            </SubmitButton>
            <FormMessage message={searchParams} />
          </div>
          <p className="text-sm text-foreground">
            Don't have an account?{" "}
            <Link className="text-primary font-medium underline" href="/sign-up">
              Sign up
            </Link>
          </p>
        </form>
      </div>
    </div>
  )
}
