document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll(".collapsible-code").forEach(container => {
        const button = container.querySelector("button");
        const pre    = container.querySelector("pre");

        pre.style.display = "none";

        button.addEventListener("click", () => {
            const isOpen = pre.style.display === "block";
            pre.style.display = isOpen ? "none" : "block";
            button.classList.toggle("expanded", !isOpen);
        });
    });
});