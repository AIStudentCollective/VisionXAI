export default function HomePage() {
    return (
            <>
    <section className="flex flex-col justify-end px-6 sm:px-8 md:px-12 lg:px-20 xl:px-28 pb-12 sm:pb-20 md:pb-24 pt-16 sm:pt-20 md:pt-24 lg:pt-32 w-full overflow-hidden">
    <div className="flex flex-col lg:flex-row justify-between items-start lg:items-end w-full mt-8 sm:mt-12 md:mt-20 lg:mt-24 gap-10 md:gap-12">
    
    {/* About Section (45%) */}
    <div className="text-white mt-5 w-full lg:w-[45%]">
            <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold text-white leading-tight mb-6">
              VisX{" "}
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-indigo-400">
                Mission
              </span>
            </h1>
    <p className="text-sm sm:text-base md:text-lg font-light leading-relaxed text-gray-200 max-w-prose">
    VisX advances the field of medical AI research by providing a transparent, interactive platform for
    interpreting vision model outputs. Through intuitive visualizations and expert-level, LLM-driven
    explanations, we enable researchers and clinicians to explore and evaluate model behavior, uncovering
    insights that drive innovation and improve model reliability. Our mission is to foster trust and
    collaboration between AI systems and medical experts, accelerating the development of safe, effective
    diagnostic tools.
    </p>
    </div>
    
    {/* Empty Space (20%) */}
    <div className="hidden lg:block lg:w-[10%] xl:w-[20%]"></div>
    
    {/* Framework Selection (35%) */}
    <div className="w-full lg:w-[35%] flex flex-col mt-10 lg:mt-0 mb-10 sm:mb-14 md:mb-20">
    <h3 className="text-2xl sm:text-3xl md:text-4xl font-bold text-white mb-4 sm:mb-5">
    Join The Community
    </h3>
    <div className="flex flex-col sm:flex-row gap-4 sm:gap-5">
    <a
    href="/sign-in"
    className="w-full sm:w-auto min-w-[8rem] md:min-w-[10rem] px-5 py-3 text-sm md:text-base font-medium text-center bg-gradient-to-r from-purple-600 to-indigo-500 rounded-xl shadow-lg hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition"
    >
    Sign In
    </a>
    <a
    href="/sign-up"
    className="w-full sm:w-auto min-w-[8rem] md:min-w-[10rem] px-5 py-3 text-sm md:text-base font-medium text-center bg-gradient-to-r from-purple-600 to-indigo-500 rounded-xl shadow-lg hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition"
    >
    Sign Up
    </a>
    </div>
    </div>
    </div>
    </section>
    </>)
    
};