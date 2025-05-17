"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { Menu, X } from "lucide-react"
import { usePathname } from "next/navigation"

export default function NavBar() {
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const [scrolled, setScrolled] = useState(false)
  const pathname = usePathname()

  // Handle scroll effect
  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 10)
    }

    window.addEventListener("scroll", handleScroll)
    return () => window.removeEventListener("scroll", handleScroll)
  }, [])

  // Close mobile menu when route changes
  useEffect(() => {
    setIsMenuOpen(false)
  }, [pathname])

  return (
    <nav
      className={`fixed top-0 pt-5 left-0 w-full z-50 transition-all duration-300 ${
        scrolled ? "bg-[#161918]/90 backdrop-blur-sm shadow-md" : "bg-transparent"
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-10">
        <div className="flex justify-between items-center h-20">
          {/* Logo */}
          <div className="flex-shrink-0 select-none">
            <Link className="select-none" draggable="false" href="/">
              <img src="/images/logo.svg" alt="Logo" className="h-16 w-auto select-none" draggable="false"/>
            </Link>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-8">
            <Link
              href="/"
              className={`font-thin hover:text-gray-300 transition-colors ${
                pathname === "/" ? "text-white" : "text-gray-400"
              }`}
            >
              Home
            </Link>
            <Link
              href="/documents"
              className={`font-thin hover:text-gray-300 transition-colors ${
                pathname === "/documents" ? "text-white" : "text-gray-400"
              }`}
            >
              Documents
            </Link>
            <Link
              href="/support"
              className={`font-thin hover:text-gray-300 transition-colors ${
                pathname === "/support" ? "text-white" : "text-gray-400"
              }`}
            >
              Support
            </Link>
            <Link
              href="/privacy-policy"
              className={`font-thin hover:text-gray-300 transition-colors ${
                pathname === "/privacy-policy" ? "text-white" : "text-gray-400"
              }`}
            >
              Privacy Policy
            </Link>
          </div>

          {/* Mobile Menu Button */}
          <div className="md:hidden flex items-center">
            <button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="inline-flex items-center justify-center p-2 rounded-md text-gray-400 hover:text-white hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-white"
              aria-expanded={isMenuOpen}
            >
              <span className="sr-only">Open main menu</span>
              {isMenuOpen ? (
                <X className="block h-6 w-6" aria-hidden="true" />
              ) : (
                <Menu className="block h-6 w-6" aria-hidden="true" />
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu */}
      <div
        className={`md:hidden transition-all duration-300 ease-in-out overflow-hidden ${
          isMenuOpen ? "max-h-64 opacity-100" : "max-h-0 opacity-0"
        }`}
      >
        <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3 bg-[#161918]/95 backdrop-blur-sm">
          <Link
            href="/"
            className={`block px-3 py-2 rounded-md text-base font-thin ${
              pathname === "/" ? "text-white bg-gradient-to-r from-purple-600 to-indigo-500" : "text-gray-300 hover:bg-purple-600 hover:bg-opacity-25 hover:text-white"
            }`}
          >
            Home
          </Link>
          <Link
            href="/documents"
            className={`block px-3 py-2 rounded-md text-base font-thin ${
              pathname === "/documents" ? "text-white bg-gradient-to-r from-purple-600 to-indigo-500" : "text-gray-300 hover:bg-purple-600 hover:bg-opacity-25 hover:text-white"
            }`}
          >
            Documents
          </Link>
          <Link
            href="/support"
            className={`block px-3 py-2 rounded-md text-base font-thin ${
              pathname === "/support" ? "text-white bg-gradient-to-r from-purple-600 to-indigo-500" : "text-gray-300 hover:bg-purple-600 hover:bg-opacity-25 hover:text-white"
            }`}
          >
            Support
          </Link>
          <Link
            href="/privacy-policy"
            className={`block px-3 py-2 rounded-md text-base font-thin ${
              pathname === "/privacy-policy"
                ? "text-white bg-gradient-to-r from-purple-600 to-indigo-500"
                : "text-gray-300 hover:bg-purple-600 hover:bg-opacity-25 hover:text-white"
            }`}
          >
            Privacy Policy
          </Link>
        </div>
      </div>
    </nav>
  )
}