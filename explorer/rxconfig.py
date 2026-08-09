import reflex as rx

config = rx.Config(
    app_name="explorer",
    plugins=[
        rx.plugins.TailwindV4Plugin(),
        rx.plugins.RadixThemesPlugin(
            theme=rx.theme(appearance="inherit", accent_color="blue",
                           radius="large"),
        ),
        rx.plugins.SitemapPlugin(),
    ],
    # Served to whatever address asks. backend_host already defaults to
    # 0.0.0.0; these are the two checks that would otherwise turn away a
    # request whose Host or Origin header is not the one we started with.
    # True rather than a list of names, because the hostname a tunnel or a
    # DHCP lease hands out is not known before it hands it out.
    vite_allowed_hosts=True,
    cors_allowed_origins=["*"],
    show_built_with_reflex=False,
)
