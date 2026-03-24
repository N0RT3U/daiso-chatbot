document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector("#chat-form");
  const input = document.querySelector("#message");
  const chatLog = document.querySelector("#chat-log");
  const submitButton = form?.querySelector(".composer-submit");

  const scrollToBottom = () => {
    window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
  };

  const setSubmitting = (isSubmitting) => {
    if (!submitButton || !input) return;
    submitButton.disabled = isSubmitting;
    input.disabled = isSubmitting;
    form?.classList.toggle("is-loading", isSubmitting);
  };

  const appendSystemMessage = (message) => {
    if (!chatLog) return;
    chatLog.insertAdjacentHTML(
      "beforeend",
      `
        <section class="exchange">
          <div class="message-row assistant">
            <div class="message-bubble assistant-bubble">
              <p class="assistant-summary">${message}</p>
            </div>
          </div>
        </section>
      `,
    );
    scrollToBottom();
  };

  const submitMessage = async (rawMessage, options = {}) => {
    const { clearInput = true, focusInput = true } = options;
    const message = (rawMessage || "").trim();
    if (!message) {
      input?.focus();
      return;
    }

    setSubmitting(true);

    try {
      const response = await fetch("/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        },
        body: new URLSearchParams({ message }).toString(),
      });

      if (!response.ok) {
        throw new Error("request failed");
      }

      const html = await response.text();
      chatLog?.insertAdjacentHTML("beforeend", html);

      if (input && clearInput) {
        input.value = "";
      }
      if (input && focusInput) {
        input.focus();
      }

      scrollToBottom();
    } catch (error) {
      appendSystemMessage("요청을 처리하지 못했습니다. 잠시 후 다시 시도해 주세요.");
    } finally {
      setSubmitting(false);
    }
  };

  document.querySelectorAll("[data-prompt]").forEach((button) => {
    button.addEventListener("click", () => {
      submitMessage(button.dataset.prompt || "", { clearInput: false, focusInput: true });
    });
  });

  form?.addEventListener("submit", async (event) => {
    event.preventDefault();
    await submitMessage(input?.value || "", { clearInput: true, focusInput: true });
  });
});
