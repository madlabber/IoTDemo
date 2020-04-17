# KVMHost.ovf

This ovf creates a VM that can be used in place of the aggregation point edge device.

- Deploy the ovf

- Install Centos 7.8 server (minimal)

- Assign a static IPv4 and hostname during install

- Run the kvmhost_prep.yml playbook against it.

- When using the virtual KVM host, use these values in the kvmhost_prep playbook:
  - virsh_pool_dev: "/dev/sdb"
  - eth_dev: "ens192"

