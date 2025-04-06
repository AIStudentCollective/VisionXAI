import DeployButton from "@/components/deploy-button";
import { EnvVarWarning } from "@/components/env-var-warning";
import HeaderAuth from "@/components/header-auth";
import { ThemeSwitcher } from "@/components/theme-switcher";
import { hasEnvVars } from "@/utils/supabase/check-env-vars";
import { Geist } from "next/font/google";
import { ThemeProvider } from "next-themes";
import Link from "next/link";
import "./globals.css";

const defaultUrl = process.env.VERCEL_URL
  ? `https://${process.env.VERCEL_URL}`
  : "http://localhost:3000";

export const metadata = {
  metadataBase: new URL(defaultUrl),
  title: "Next.js and Supabase Starter Kit",
  description: "The fastest way to build apps with Next.js and Supabase",
};

const geistSans = Geist({
  display: "swap",
  subsets: ["latin"],
});

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={geistSans.className} suppressHydrationWarning>
      <body className="bg-black text-white text-foreground min-h-screen min-w-full flex flex-col">
        
        {/* Background Wrapper */}
        <div className="relative min-h-screen w-full flex flex-col bg-[url('/images/background.svg')] bg-cover bg-center bg-no-repeat">
          
          {/* Navbar */}
          <div className="absolute top-0 left-0 w-full flex justify-between px-10 py-6">
            <img src="/images/logo.svg" alt="Logo" className="w-60 h-60"/>
            <div className="flex flex-row p-5 mt-20"> 
              <Link href="/" className="mx-5">Home</Link>
              <Link href="" className="mx-5">Documents</Link>
              <Link href="" className="mx-5">Support</Link>
              <Link href="" className="mx-5">Privacy Policy</Link>
            </div>
          </div>

          {/* Page Content - Adding Padding to Push Below Navbar */}
          <div className="flex-1 flex items-center justify-center w-full h-full pt-[8rem]">
            {children}
          </div>

        </div>
        
      </body>
    </html>
  );
}
