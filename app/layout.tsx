import DeployButton from "@/components/deploy-button";
import { EnvVarWarning } from "@/components/env-var-warning";
import HeaderAuth from "@/components/header-auth";
import { ThemeSwitcher } from "@/components/theme-switcher";
import { hasEnvVars } from "@/utils/supabase/check-env-vars";
import { Geist } from "next/font/google";
import { ThemeProvider } from "next-themes";
import Link from "next/link";
import "./globals.css";
import NavBar from "@/components/Navbar";

const defaultUrl = process.env.VERCEL_URL
? `https://${process.env.VERCEL_URL}`
: "http://localhost:3000";

export const metadata = {
	metadataBase: new URL(defaultUrl),
	title: "VisX",
	description: "",
  icons: {
    icon: '/favicon.ico',
  },
};

const geistSans = Geist({
	display: "swap",
	subsets: ["latin"],
});

import { Inter } from 'next/font/google';

const inter = Inter({ 
	// display: 'swap ',
	subsets: ['latin'] 
});

export default function RootLayout({
	children,
}: Readonly<{
	children: React.ReactNode;
}>) {
	return (
		<html lang= "en" className={inter.className} suppressHydrationWarning>
			<body className="bg-[#161918] text-white text-foreground min-h-screen min-w-full flex flex-col">
				{/* Navbar */}
				<NavBar/>
				{/* Page Content - Adding Padding to Push Below Navbar */}
				<div className="relative flex-1 flex items-center justify-center w-fit-content h-fit-content bg-[url('/images/background.svg')] bg-cover bg-center bg-no-repeat">
				{children}
				</div>
				<footer className="bg-transparent text-white w-full flex flex-col items-center justify-center">
						<p className="text-xs my-2">VisX @ AI Student Collective</p>
				</footer>
			</body>
		</html>
	);
}