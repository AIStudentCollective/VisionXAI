import type React from "react"
import { Card } from "@/components/ui/card"
import { Code, Palette, Briefcase } from "lucide-react"

interface TeamMember {
  name: string
  role: string
  description: string
  image: string
  icon: React.ReactNode
}

export default function AboutPage() {
  const teamMembers: TeamMember[] = [
    {
      name: "Alex Morgan",
      role: "Project Manager",
      description: "Strategic visionary who guides our team toward excellence.",
      image: "/placeholder.svg?height=300&width=300",
      icon: <Briefcase className="h-5 w-5" />,
    },
    {
      name: "Sophia Chen",
      role: "Lead Designer",
      description: "Creative genius with an eye for stunning visual experiences.",
      image: "/placeholder.svg?height=300&width=300",
      icon: <Palette className="h-5 w-5" />,
    },
    {
      name: "Marcus Lee",
      role: "UI/UX Designer",
      description: "Passionate about creating intuitive and accessible interfaces.",
      image: "/placeholder.svg?height=300&width=300",
      icon: <Palette className="h-5 w-5" />,
    },
    {
      name: "Olivia Rodriguez",
      role: "Frontend Developer",
      description: "Crafts pixel-perfect implementations of our design vision.",
      image: "/placeholder.svg?height=300&width=300",
      icon: <Code className="h-5 w-5" />,
    },
    {
      name: "James Wilson",
      role: "Backend Developer",
      description: "Architects robust systems that power our innovative solutions.",
      image: "/placeholder.svg?height=300&width=300",
      icon: <Code className="h-5 w-5" />,
    },
    {
      name: "Aisha Patel",
      role: "Full Stack Developer",
      description: "Versatile coder who bridges the gap between frontend and backend.",
      image: "/placeholder.svg?height=300&width=300",
      icon: <Code className="h-5 w-5" />,
    },
    {
      name: "David Kim",
      role: "DevOps Engineer",
      description: "Ensures our infrastructure runs smoothly and securely.",
      image: "/placeholder.svg?height=300&width=300",
      icon: <Code className="h-5 w-5" />,
    },
  ]

  return (
    <div className="container mx-auto px-4 py-24 md:py-32">
      <div className="text-center mb-16">
        <h1 className="text-4xl md:text-5xl font-bold mb-6 bg-gradient-to-r from-gray-200 to-gray-400 bg-clip-text text-transparent">
          Meet Our Team
        </h1>
        <p className="text-gray-400 max-w-2xl mx-auto text-lg">
          The talented individuals behind VisX who are passionate about creating exceptional experiences.
        </p>
      </div>

      {/* Team Categories */}
      <div className="space-y-16">
        {/* Project Management */}
        <div>
          <h2 className="text-2xl font-semibold mb-8 text-gray-300 border-b border-gray-800 pb-2">
            Project Management
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-1 gap-8">
            {teamMembers
              .filter((member) => member.role.includes("Project Manager"))
              .map((member, index) => (
                <TeamMemberCard key={index} member={member} />
              ))}
          </div>
        </div>

        {/* Design */}
        <div>
          <h2 className="text-2xl font-semibold mb-8 text-gray-300 border-b border-gray-800 pb-2">Design</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {teamMembers
              .filter((member) => member.role.includes("Designer"))
              .map((member, index) => (
                <TeamMemberCard key={index} member={member} />
              ))}
          </div>
        </div>

        {/* Development */}
        <div>
          <h2 className="text-2xl font-semibold mb-8 text-gray-300 border-b border-gray-800 pb-2">Development</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {teamMembers
              .filter((member) => member.role.includes("Developer") || member.role.includes("Engineer"))
              .map((member, index) => (
                <TeamMemberCard key={index} member={member} />
              ))}
          </div>
        </div>
      </div>
    </div>
  )
}

function TeamMemberCard({ member }: { member: TeamMember }) {
  return (
    <Card className="overflow-hidden bg-[#1e211f] border-gray-800 hover:border-gray-700 transition-all duration-300 hover:shadow-lg hover:shadow-emerald-900/10 group">
      <div className="relative">
        <div className="aspect-square overflow-hidden bg-gradient-to-br from-gray-900 to-gray-800">
          <img
            src={member.image || "/placeholder.svg"}
            alt={member.name}
            className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
          />
        </div>
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4">
          <div className="flex items-center space-x-2">
            <div className="p-1.5 rounded-full bg-emerald-900/50 text-emerald-400">{member.icon}</div>
            <span className="text-sm font-medium text-emerald-400">{member.role}</span>
          </div>
        </div>
      </div>
      <div className="p-5">
        <h3 className="text-xl font-semibold mb-2">{member.name}</h3>
        <p className="text-gray-400">{member.description}</p>
      </div>
    </Card>
  )
}
