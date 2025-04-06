
import Hero from "@/components/hero";
import ConnectSupabaseSteps from "@/components/tutorial/connect-supabase-steps";
import SignUpUserSteps from "@/components/tutorial/sign-up-user-steps";
import { hasEnvVars } from "@/utils/supabase/check-env-vars";

export default async function Home() {
  return (
    <>
      <section className="flex flex-col justify-end px-16 pb-20 pt-[6rem]">
        <div className="flex justify-end items-end w-full">
          {/* About Section (45%) */}
          <div className="text-white w-[45%] ml-6">
            <h2 className="text-[48px] font-semibold mb-12">About</h2>
            <p className="text-[20px] font-normal">
              Lorem ipsum dolor sit amet. Eos fugiat saepe et obcaecati deserunt non enim fuga aut esse voluptas ut totam dolor 
              hic natus quasi sed earum Quis. Et Quis officiis ea molestiae quae sit maiores reiciendis rem dicta repellat! 
              Ea quia quidem et repellendus aperiam sit voluptatum atque in neque repudiandae a dignissimos voluptate est dolorem 
              dolores aut fugiat dolore. Ut magni dolorem sed blanditiis molestiae qui atque deserunt.
            </p>
            <p className="text-[20px] font-normal mt-6">
              Lorem ipsum dolor sit amet. Eos fugiat saepe et obcaecati deserunt non enim fuga aut esse voluptas ut totam 
              dolor hic natus quasi sed earum Quis. Et Quis officiis ea molestiae quae sit maiores reiciendis rem dicta repellat!
            </p>
          </div>

          {/* Empty Space (20%) */}
          <div className="w-[20%]"></div>

          {/* Framework Selection (35%) */}
          <div className="w-[35%] flex flex-col mb-20">
            <h3 className="text-[28px] font-semibold text-white mb-4">
              Select A Framework
            </h3>
            <div className="flex gap-4">
              <a
                href="/torch"
                className="w-[10rem] px-8 py-3 text-[16px] font-medium text-center bg-[linear-gradient(90deg,#9333EA_0%,#6366F1_100%)] rounded-xl shadow-md hover:opacity-80 transition-all"
              >
                PyTorch
              </a>
              <a
                href="/tensorflow"
                className=" w-[10rem] px-8 py-3 text-[16px] font-medium text-center bg-[linear-gradient(90deg,#9333EA_0%,#6366F1_100%)] rounded-xl shadow-md hover:opacity-80 transition-all"
              >
                TensorFlow
              </a>
            </div>
          </div>
        </div>
      </section>
    </>
  );
}
