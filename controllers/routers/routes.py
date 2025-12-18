from fastapi import APIRouter, Depends

# from controllers.auth.auth_handler import check_auth
from controllers.routers import eval_router, analysis_router, dataset_router, completions_router, results_router, evaluation_configs, \
    project_router, cache_upsert

router = APIRouter(
    # dependencies=[Depends(check_auth)],
    prefix='/evaluation'
)

router.include_router(eval_router.router)
router.include_router(analysis_router.router)
router.include_router(dataset_router.router)
router.include_router(completions_router.router)
router.include_router(results_router.router)
router.include_router(project_router.router)
router.include_router(evaluation_configs.router)
router.include_router(cache_upsert.router)