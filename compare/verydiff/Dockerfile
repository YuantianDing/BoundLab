FROM julia:1.10

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    gcc

COPY VeryDiff /VeryDiff

RUN cd /VeryDiff && \
    julia --project=. -e 'using Pkg; Pkg.instantiate()'

RUN printf '#!/bin/sh\nexec julia --project=/VeryDiff -e "using VeryDiff; VeryDiff.main_VeryDiff()" -- "$@"\n' > /usr/local/bin/VeryDiff && \
    chmod +x /usr/local/bin/VeryDiff

ENTRYPOINT ["/usr/local/bin/VeryDiff"]