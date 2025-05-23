"use client"

import { useEffect, useState } from "react"

interface TypedHeadingProps {
  text: string
  className?: string
  speed?: number
  highlightedText?: string
}

export default function TypedHeading({ text, className = "", speed = 100, highlightedText }: TypedHeadingProps) {
  const [displayText, setDisplayText] = useState("")
  const [isTyping, setIsTyping] = useState(true)
  const [isComplete, setIsComplete] = useState(false)

  useEffect(() => {
    let currentIndex = 0
    setDisplayText("")
    setIsTyping(true)
    setIsComplete(false)

    const typingInterval = setInterval(() => {
      if (currentIndex < text.length) {
        setDisplayText((prev) => prev + text.charAt(currentIndex))
        currentIndex++
      } else {
        clearInterval(typingInterval)
        setIsTyping(false)
        setIsComplete(true)
      }
    }, speed)

    return () => clearInterval(typingInterval)
  }, [text, speed])

  if (highlightedText && isComplete) {
    const parts = text.split(highlightedText)
    return (
      <h2 className={className}>
        {parts[0]}
        <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-indigo-400 font-extrabold">
          {highlightedText}
        </span>
        {parts[1]}
        {isTyping && <span className="inline-block w-[0.1em] h-[1em] bg-purple-400 ml-1 animate-blink"></span>}
      </h2>
    )
  }

  return (
    <h2 className={className}>
      {displayText}
      {isTyping && <span className="inline-block w-[0.1em] h-[1em] bg-purple-400 ml-1 animate-blink"></span>}
    </h2>
  )
}
