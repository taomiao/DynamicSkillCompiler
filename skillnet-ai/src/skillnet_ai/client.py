import os
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

from skillnet_ai.creator import SkillCreator
from skillnet_ai.downloader import SkillDownloader
from skillnet_ai.evaluator import SkillEvaluator, EvaluatorConfig
from skillnet_ai.searcher import SkillNetSearcher
from skillnet_ai.analyzer import SkillRelationshipAnalyzer
from skillnet_ai.compiler import (
    CompilerConfig,
    CompositeSkillRetriever,
    DynamicSkillCompiler,
    LocalEnvironment,
    LocalSkillLibraryRetriever,
    SkillNetSearchRetriever,
)

class SkillNetError(Exception):
    """Custom exception class for SkillNet Client errors."""
    pass

class SkillNetClient:
    """
    A Python SDK client for interacting with SkillNet services.
    
    This client aggregates Search, Download, Create, Evaluate, and Analyze functionalities.
    """

    def __init__(
        self, 
        api_key: Optional[str] = None, 
        base_url: Optional[str] = None,
        github_token: Optional[str] = None
    ):
        """
        Initialize the SkillNet Client.

        Args:
            api_key: OpenAI/SkillNet API Key. Defaults to env var API_KEY.
            base_url: Base URL for the LLM API. Defaults to env var BASE_URL or OpenAI default.
            github_token: GitHub token for downloading private skills or avoiding rate limits.
                          Defaults to env var GITHUB_TOKEN.
        """
        self.api_key = api_key or os.getenv("API_KEY")
        self.base_url = base_url or os.getenv("BASE_URL")
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")


    def search(
        self,
        q: str,
        mode: str = "keyword",
        category: Optional[str] = None,
        limit: int = 20,
        page: int = 1,
        min_stars: int = 0,
        sort_by: str = "stars",
        threshold: float = 0.8
    ) -> List[Any]:
        """
        Search for skills on SkillNet.

        Args:
            q: The search query.
            mode: 'keyword' or 'vector'.
            category: Filter by category.
            limit: Max results.
            page: Page number (keyword mode only).
            min_stars: Filter by stars (keyword mode only).
            sort_by: 'stars' or 'recent' (keyword mode only).
            threshold: Similarity threshold (vector mode only).

        Returns:
            A list of skill objects found.
        """
        try:
            searcher = SkillNetSearcher()
            results = searcher.search(
                q=q,
                mode=mode, 
                category=category,
                limit=limit,
                page=page,
                min_stars=min_stars,
                sort_by=sort_by,
                threshold=threshold
            )
            return results
        except Exception as e:
            raise SkillNetError(f"Search failed: {str(e)}") from e

    def download(
        self,
        url: str,
        target_dir: str = ".",
        token: Optional[str] = None
    ) -> str:
        """
        Download a skill from a GitHub URL.

        Args:
            url: The GitHub URL of the specific skill folder.
            target_dir: Local directory to install into.
            token: Optional override for GitHub token.

        Returns:
            str: The absolute path to the installed skill folder.

        Raises:
            SkillNetError: If download fails.
        """
        # Use instance token if specific token not provided
        use_token = token if token else self.github_token
        downloader = SkillDownloader(api_token=use_token)

        try:
            installed_path = downloader.download(folder_url=url, target_dir=target_dir)
            if not installed_path:
                raise SkillNetError("Download returned None. Check URL validity or permissions.")
            return os.path.abspath(installed_path)
        except Exception as e:
            raise SkillNetError(f"Download failed: {str(e)}") from e

    def create(
        self,
        input_type: str = "auto",
        trajectory_content: Optional[str] = None,
        github_url: Optional[str] = None,
        office_file: Optional[str] = None,
        prompt: Optional[str] = None,
        output_dir: Union[str, Path] = "./generated_skills",
        model: str = "gpt-4o",
        max_files: int = 50
    ) -> List[str]:
        """
        Generate executable skills from various input sources.

        Args:
            input_type: Input source type. One of:
                - "auto": Auto-detect based on provided parameters (default)
                - "github": Create from GitHub repository
                - "trajectory": Create from execution log/trajectory
                - "office": Create from PDF/PPT/Word document
                - "prompt": Create from user's direct description
            trajectory_content: The text content of the execution log/trajectory.
            github_url: Full URL to GitHub repository.
            office_file: Path to office document (PDF, PPT, Word).
            prompt: User's description for prompt-based skill creation.
            output_dir: Directory where new skills will be saved.
            model: The LLM model to use.
            max_files: Maximum code files to analyze (GitHub mode only).

        Returns:
            List[str]: A list of paths to the generated skill folders.
        """
        if not self.api_key:
            raise SkillNetError("API_KEY is required for skill creation.")

        # Auto-detect input type if not specified
        if input_type == "auto":
            if github_url:
                input_type = "github"
            elif trajectory_content:
                input_type = "trajectory"
            elif office_file:
                input_type = "office"
            elif prompt:
                input_type = "prompt"
            else:
                raise SkillNetError(
                    "Must provide one of: trajectory_content, github_url, "
                    "office_file, or diy_prompt."
                )

        # Validate input_type
        valid_types = {"github", "trajectory", "office", "prompt", "auto"}
        if input_type not in valid_types:
            raise SkillNetError(f"Invalid input_type: {input_type}. Must be one of {valid_types}")

        try:
            creator = SkillCreator(
                api_key=self.api_key, 
                base_url=self.base_url, 
                model=model
            )
            
            if input_type == "github":
                if not github_url:
                    raise SkillNetError("github_url is required for github input type.")
                created_paths = creator.create_from_github(
                    github_url=github_url,
                    output_dir=str(output_dir),
                    api_token=self.github_token,
                    max_files=max_files
                )
            elif input_type == "trajectory":
                if not trajectory_content:
                    raise SkillNetError("trajectory_content is required for trajectory input type.")
                created_paths = creator.create_from_trajectory(
                    trajectory=trajectory_content,
                    output_dir=str(output_dir)
                )
            elif input_type == "office":
                if not office_file:
                    raise SkillNetError("office_file is required for office input type.")
                created_paths = creator.create_from_office(
                    file_path=office_file,
                    output_dir=str(output_dir)
                )
            elif input_type == "prompt":
                if not prompt:
                    raise SkillNetError("prompt is required for prompt input type.")
                created_paths = creator.create_from_prompt(
                    user_input=prompt,
                    output_dir=str(output_dir)
                )
            else:
                raise SkillNetError(f"Unknown input_type: {input_type}")
            
            return created_paths if created_paths else []
        except Exception as e:
            raise SkillNetError(f"Creation failed: {str(e)}") from e

    def evaluate(
        self,
        target: str,
        name: Optional[str] = None,
        category: Optional[str] = None,
        description: Optional[str] = None,
        model: str = "gpt-4o",
        max_workers: int = 5,
        cache_dir: Union[str, Path] = "./evaluate_cache_dir"
    ) -> Dict[str, Any]:
        """
        Evaluate a skill (local path or URL).

        Args:
            target: Local folder path OR GitHub URL.
            name: Override skill name.
            category: Override skill category.
            description: Override skill description.
            model: LLM model for evaluation.
            max_workers: Concurrency limit.

        Returns:
            Dict[str, Any]: The evaluation report dictionary.
        """
        if not self.api_key:
            raise SkillNetError("API_KEY is required for evaluation.")

        config = EvaluatorConfig(
            api_key=self.api_key,
            base_url=self.base_url,
            model=model,
            max_workers=max_workers,
            cache_dir=cache_dir,
            github_token=self.github_token
        )
        evaluator = SkillEvaluator(config)

        try:
            is_url = target.startswith("http://") or target.startswith("https://")
            
            if is_url:
                result = evaluator.evaluate_from_url(
                    url=target, 
                    name=name, 
                    category=category, 
                    description=description
                )
            else:
                result = evaluator.evaluate_from_path(
                    path=target, 
                    name=name, 
                    category=category, 
                    description=description
                )
            
            if "error" in result:
                raise SkillNetError(f"Evaluation logic returned error: {result['error']}")
                
            return result

        except Exception as e:
            raise SkillNetError(f"Evaluation process failed: {str(e)}") from e

    def analyze(
        self,
        skills_dir: Union[str, Path],
        save_to_file: bool = True,
        model: str = "gpt-4o"
    ) -> List[Dict[str, Any]]:
        """
        Analyze a local directory containing multiple skills to infer relationships between them.
        
        This builds a knowledge graph (edges) between skills based on their names and descriptions.
        Relationships detected: similar_to, belong_to, compose_with, depend_on.

        Args:
            skills_dir: Path to the directory containing skill folders.
            save_to_file: If True, saves a 'relationships.json' file in the skills_dir.
            model: The LLM model to use for analysis.

        Returns:
            List[Dict[str, Any]]: A list of relationship edges (source, target, type, reason).
        """
        if not self.api_key:
            raise SkillNetError("API_KEY is required for relationship analysis.")

        try:
            analyzer = SkillRelationshipAnalyzer(
                api_key=self.api_key,
                base_url=self.base_url,
                model=model
            )
            
            results = analyzer.analyze_local_skills(
                skills_dir=str(skills_dir),
                save_to_file=save_to_file
            )
            return results
        except Exception as e:
            raise SkillNetError(f"Relationship analysis failed: {str(e)}") from e

    def compile_skills(
        self,
        query: str,
        local_skills_dir: Optional[Union[str, Path]] = None,
        search_limit: int = 20,
        vector_threshold: float = 0.7,
        min_relevance: float = 0.25,
        preserve_top_k: int = 2,
        similar_prune_margin: float = 0.08,
        keep_parent_if_better_by: float = 0.05,
        coverage_weight: float = 0.55,
        quality_weight: float = 0.20,
        cost_weight: float = 0.15,
        latency_weight: float = 0.10,
        cwd: Optional[Union[str, Path]] = None,
        workspace_root: Optional[Union[str, Path]] = None,
        python_bin: str = "python",
        shell: str = "sh",
        os_name: str = "unknown",
        available_bins: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compile a task-specific, de-duplicated skill package for the given query.

        This method combines local skill-library retrieval with SkillNet search retrieval,
        constructs a multi-relational skill graph, then prunes and localizes it for the
        current environment.
        """
        retrievers = []
        if local_skills_dir:
            retrievers.append(LocalSkillLibraryRetriever(str(local_skills_dir)))
        retrievers.append(
            SkillNetSearchRetriever(
                searcher=SkillNetSearcher(),
                limit=search_limit,
                threshold=vector_threshold,
            )
        )

        compiler = DynamicSkillCompiler(
            retriever=CompositeSkillRetriever(retrievers=retrievers),
            config=CompilerConfig(
                min_relevance=min_relevance,
                preserve_top_k=preserve_top_k,
                similar_prune_margin=similar_prune_margin,
                keep_parent_if_better_by=keep_parent_if_better_by,
                coverage_weight=coverage_weight,
                quality_weight=quality_weight,
                cost_weight=cost_weight,
                latency_weight=latency_weight,
            ),
        )
        environment = LocalEnvironment(
            cwd=str(cwd or os.getcwd()),
            workspace_root=str(workspace_root or os.getcwd()),
            python_bin=python_bin,
            shell=shell,
            os_name=os_name,
            available_bins=set(available_bins or []),
        )
        compiled = compiler.compile(query=query, environment=environment)

        return {
            "summary": DynamicSkillCompiler.summarize(compiled),
            "subgoals": [
                {
                    "subgoal_id": subgoal.subgoal_id,
                    "description": subgoal.description,
                    "required_capabilities": sorted(subgoal.required_capabilities),
                    "depends_on": subgoal.depends_on,
                    "environment_hints": subgoal.environment_hints,
                }
                for subgoal in compiled.subgoals
            ],
            "execution_order": compiled.execution_order,
            "selected_skills": [
                {
                    "skill_id": item.asset.skill_id,
                    "name": item.asset.name,
                    "location": item.asset.location,
                    "utility_score": item.utility_score,
                    "assigned_subgoals": item.assigned_subgoals,
                    "selected_fragments": [
                        {
                            "fragment_id": fragment.fragment_id,
                            "content": fragment.content,
                            "example_actions": fragment.example_actions,
                        }
                        for fragment in item.selected_fragments
                    ],
                    "localized_instructions": item.localized_instructions,
                    "selected_reason": item.selected_reason,
                }
                for item in compiled.compiled_skills
            ],
            "relations": [
                {
                    "source": relation.source,
                    "target": relation.target,
                    "type": relation.relation_type,
                    "weight": relation.weight,
                    "reason": relation.reason,
                }
                for relation in compiled.graph.relations
            ],
            "metrics": compiled.metrics.__dict__,
            "dropped_skills": compiled.dropped_skills,
            "notes": compiled.notes,
        }
