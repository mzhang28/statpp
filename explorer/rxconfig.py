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
)
