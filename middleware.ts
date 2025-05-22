import { createMiddlewareClient } from "@supabase/auth-helpers-nextjs"
import { NextResponse } from "next/server"
import type { NextRequest } from "next/server"

// List of paths that require authentication
const protectedPaths = ["/dashboard", "/database", "/torch_vit", "/tensorflow"]

// List of paths that should redirect to dashboard if user is already authenticated
const authPaths = ["/sign-in", "/sign-up", "/forgot-password"]

export async function middleware(request: NextRequest) {
  const response = NextResponse.next()
  const supabase = createMiddlewareClient({ req: request, res: response })

  const {
    data: { session },
  } = await supabase.auth.getSession()

  const url = new URL(request.url)
  const path = url.pathname

  // If the user is on a protected path and not authenticated, redirect to sign-in
  if (protectedPaths.some((protectedPath) => path.startsWith(protectedPath)) && !session) {
    const redirectUrl = new URL("/sign-in", request.url)
    redirectUrl.searchParams.set("redirect", path)
    return NextResponse.redirect(redirectUrl)
  }

  // If the user is on an auth path and is authenticated, redirect to dashboard
  if (authPaths.some((authPath) => path === authPath) && session) {
    return NextResponse.redirect(new URL("/dashboard", request.url))
  }

  return response
}

// Only run middleware on specific paths
export const config = {
  matcher: [
    "/dashboard/:path*",
    "/database/:path*",
    "/torch_vit/:path*",
    "/tensorflow/:path*",
    "/sign-in",
    "/sign-up",
    "/forgot-password",
  ],
}
