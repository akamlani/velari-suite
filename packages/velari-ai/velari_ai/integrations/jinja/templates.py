import  jinja2 as j2
import  textwrap
from    pathlib import Path
from    typing  import Any, Dict, Optional, Tuple



class TemplateJ2(object):
    """Load and render Jinja2 prompt template files from a directory.

    Configures the environment to strip leading whitespace and trailing newlines, and
    exposes `isinstance`, `hasattr`, `type`, and `len` as global functions within
    templates. When `template_path` is `None`, no environment is created — only
    `render_from_string()` is usable; the file-based methods raise `RuntimeError`.

    Args:
        template_path (Optional[str]): Directory containing the Jinja2 template files.
        **kwargs (Any): Reserved for future extension; currently unused.

    Examples:
        >>> prompt_exec = TemplateJ2(template_path=exp.conf_dir.joinpath("prompts"))
        >>> template: j2.Template = prompt_exec.load_template("child.j2")
        >>> output: str = prompt_exec.render("child.j2", content={"name": "John"})
    """
    def __init__(self, template_path: Optional[str] = None, **kwargs: Any) -> None:
        self.template_path: Optional[Path] = None
        self.env: Optional[j2.Environment] = None
        if template_path is not None:
            self.template_path = Path(template_path)
            self.env = j2.Environment(
                loader=j2.FileSystemLoader(self.template_path),
                trim_blocks=True,
                lstrip_blocks=True,
                keep_trailing_newline=True,
                undefined=j2.DebugUndefined,
            )
            # make some python core functionality available in template files
            self.env.globals["isinstance"] = isinstance
            self.env.globals["hasattr"]    = hasattr
            self.env.globals["type"]       = type
            self.env.globals["len"]        = len

    def _require_env(self) -> Tuple[Path, j2.Environment]:
        if self.template_path is None or self.env is None:
            raise RuntimeError(
                "TemplateJ2 was constructed without template_path — file-based methods are "
                "unavailable; use render_from_string() instead"
            )
        return self.template_path, self.env

    def load_text(self, file_path: str) -> str:
        """Load a Jinja2 template file as a raw stripped text string.

        Args:
            file_path (str): Relative path to the `*.j2` file within `template_path`.

        Returns:
            str: Contents of the file with leading and trailing whitespace removed.
        """
        template_path, _ = self._require_env()
        with open(template_path.joinpath(file_path), "r", encoding="utf-8") as file:
            return file.read().strip()

    def load_source(self, template_file: str) -> str:
        """Retrieve the Jinja2 template source string via the environment loader.

        Args:
            template_file (str): Name of the `*.j2` file to load from the template directory.

        Returns:
            str: Raw template source string with leading and trailing whitespace removed.
        """
        _, env = self._require_env()
        if env.loader is None:
            raise RuntimeError("TemplateJ2's environment has no loader configured")
        source, _, _ = env.loader.get_source(env, template_file)
        return source.strip()

    def load_template(self, template_file: str) -> j2.Template:
        """Load and return a compiled Jinja2 Template object.

        Args:
            template_file (str): Name of the `*.j2` file to compile from the template directory.

        Returns:
            j2.Template: Compiled Jinja2 template ready for rendering.
        """
        _, env = self._require_env()
        return env.get_template(template_file)

    @classmethod
    def render_from_string(cls, template_string: str, template_context: Dict[str, Any], **kwargs: Any) -> str:
        """Render a Jinja2 template string directly without a file loader.

        Args:
            template_string (str): Raw Jinja2 template source to compile and render.
            template_context (Dict[str, Any]): Variables available inside the template.
            **kwargs (Any): Additional variables merged into the render context.

        Returns:
            str: Rendered output string with all template expressions evaluated.

        Examples:
            >>> output = TemplateJ2.render_from_string("Hello, {{ name }}!", {"name": "Ada"})
        """
        template = j2.Template(template_string, trim_blocks=True, lstrip_blocks=True)
        return template.render(template_context, **kwargs)

    def render(self, template_file: str, content: Dict[str, Any], **kwargs: Any) -> str:
        """Render a Jinja2 template file with the given content dictionary.

        Args:
            template_file (str): Name of the `*.j2` file in the template directory to render.
            content (Dict[str, Any]): Variables passed into the template context.
            **kwargs (Any): Additional variables merged into the render context.

        Returns:
            str: Rendered output with trailing whitespace removed.

        Examples:
            >>> prompt_exec = TemplateJ2(template_path="config/prompts")
            >>> output = prompt_exec.render("child.j2", content={"name": "John", "age": 30})
        """
        _, env = self._require_env()
        template = env.get_template(template_file)
        return textwrap.dedent(template.render(content, **kwargs).rstrip())
