import { signOutAction } from "@/app/actions"
import { Button } from "@/components/ui/button"

export default function SignOut() {
    return (
        <Button onClick={signOutAction}>
            Sign Out
        </Button>
    )
}