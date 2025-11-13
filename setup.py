"""
Setup module for AutoHighlight FastAPI integration.
Use this in your BallQ_beta application to initialize AutoHighlight.
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Add autohighlight to Python path if running as submodule
def setup_autohighlight_path(submodule_path: Optional[str] = None):
    """
    Add AutoHighlight to Python path for module imports.
    
    Args:
        submodule_path: Path to autohighlight submodule. If None, uses relative path.
    """
    if submodule_path is None:
        # Assume we're being imported from services/autohighlight or modules/autohighlight
        submodule_path = Path(__file__).parent
    
    submodule_path = Path(submodule_path).resolve()
    if str(submodule_path) not in sys.path:
        sys.path.insert(0, str(submodule_path))


def get_autohighlight_router():
    """
    Get the FastAPI router for AutoHighlight.
    
    Returns:
        APIRouter configured with AutoHighlight endpoints
    """
    from ballq_api import router
    return router


def configure_autohighlight(
    model_path: str = "models/default.hdf5",
    submodule_path: Optional[str] = None
):
    """
    Configure AutoHighlight for use in BallQ_beta.
    Call this during application startup.
    
    Args:
        model_path: Path to the model file
        submodule_path: Path to autohighlight submodule
    """
    # Setup path
    setup_autohighlight_path(submodule_path)
    
    # Initialize API
    from fastapi_wrapper import get_autohighlight_api
    api = get_autohighlight_api(model_path)
    
    return api


# Example FastAPI app configuration
def create_app_with_autohighlight(model_path: str = "models/default.hdf5"):
    """
    Create a FastAPI app with AutoHighlight integration pre-configured.
    
    Usage:
        app = create_app_with_autohighlight()
        
    Or integrate with existing app:
        from setup import get_autohighlight_router
        app.include_router(get_autohighlight_router())
    """
    from fastapi import FastAPI
    
    app = FastAPI(
        title="AutoHighlight API",
        description="Automatic video summarizer using action recognition",
        version="1.0.0"
    )
    
    # Configure AutoHighlight
    configure_autohighlight(model_path)
    
    # Include router
    app.include_router(get_autohighlight_router())
    
    @app.on_event("startup")
    async def startup_event():
        print("AutoHighlight API initialized")
    
    return app


if __name__ == "__main__":
    # Example usage
    print("AutoHighlight setup module")
    print("Usage in BallQ_beta:")
    print("  from autohighlight.setup import configure_autohighlight, get_autohighlight_router")
    print("  configure_autohighlight()")
    print("  app.include_router(get_autohighlight_router())")
