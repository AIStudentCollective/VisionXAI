import { type NextRequest, NextResponse } from "next/server";
import { updateSession } from "@/utils/supabase/middleware";

export async function middleware(request: NextRequest) {
  // Redirect root to tensorflow-support
  if (request.nextUrl.pathname === '/') {
    return NextResponse.redirect(new URL('/tensorflow-support', request.url));
  }

  // Continue with the existing session update logic
  return await updateSession(request);
}

export const config = {
  matcher: [
    "/",
    "/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)",
  ],
};
