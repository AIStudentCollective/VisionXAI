import Hero from "@/components/hero";
import LandingPage from "@/components/LandingPage";
import ConnectSupabaseSteps from "@/components/tutorial/connect-supabase-steps";
import SignUpUserSteps from "@/components/tutorial/sign-up-user-steps";
import { hasEnvVars } from "@/utils/supabase/check-env-vars";

export const metadata = {
  title: 'VisX | Home',
};

export default async function Home() {
  return (
    <LandingPage />  
  )
}