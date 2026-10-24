"""Opt-in cached-diffusion candidate; normal inference never imports it."""
import sys
from cached_diffusion import install
from measure_inference import main


if __name__=='__main__':
    metrics=next(x.split('=',1)[1] for x in sys.argv if x.startswith('--metrics='))
    save=install(metrics.replace('resource_metrics.json','cached_diffusion.json'))
    try:main()
    finally:save()
