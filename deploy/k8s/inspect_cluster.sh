#!/usr/bin/env bash
# Read-only look at the Kubernetes cluster from the DGX login node. Changes nothing.
#   bash inspect_cluster.sh
echo "== Who and where"
echo "  user: $(id -un)   context: $(kubectl config current-context 2>/dev/null)"
NS=$(kubectl config view --minify -o jsonpath='{..namespace}' 2>/dev/null); echo "  namespace: ${NS:-default}"
echo "== Can I create...?"
for r in pods services persistentvolumeclaims; do
  printf '  %-24s %s\n' "$r" "$(kubectl auth can-i create $r 2>/dev/null)"
done
echo "== GPUs per node (allocatable)"
kubectl get nodes -o custom-columns='NODE:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu,CPU:.status.allocatable.cpu,MEM:.status.allocatable.memory' 2>&1 | sed 's/^/  /'
kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.labels.nvidia\.com/gpu\.product}{"\n"}{end}' 2>/dev/null | sort | uniq -c | sed 's/^/  GPU model: /'
echo "== Quotas and limits in my namespace"
kubectl get resourcequota,limitrange 2>&1 | sed 's/^/  /'
kubectl describe resourcequota 2>/dev/null | grep -E 'gpu|memory|cpu|storage|pods' | sed 's/^/  /'
echo "== Storage"
kubectl get storageclass 2>&1 | sed 's/^/  /'
kubectl get pvc 2>&1 | sed 's/^/  /'
echo "== My pods and services (names/ports already used)"
kubectl get pods,services -o wide 2>&1 | sed 's/^/  /'
echo "== NodePorts in use on the cluster"
kubectl get svc -A -o jsonpath='{range .items[*]}{range .spec.ports[*]}{.nodePort}{"\n"}{end}{end}' 2>/dev/null | grep -v '^$' | sort -n | tr '\n' ' ' | sed 's/^/  /'; echo
echo "== Login node"
df -h "$HOME" | awk 'NR==2{print "  home: "$4" free of "$2}'
command -v envsubst >/dev/null && echo "  envsubst: yes" || echo "  envsubst: no (deploy.sh uses sed)"
