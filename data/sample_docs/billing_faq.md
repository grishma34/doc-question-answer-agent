# Aurora Billing FAQ

This FAQ covers how teams are charged for Aurora platform usage. All
figures are fictional and exist only as sample data for the QA agent.

## How is usage billed?

Teams are billed monthly based on reserved resources, not actual
utilization. The unit of billing is the "compute unit" (CU): one CU equals
1 vCPU plus 2 GiB of memory reserved for one hour. A CU costs $0.04.

## What payment methods are accepted?

Internal teams are charged through cost-center chargeback; no payment
method is required. External partner teams can pay by corporate credit
card or by invoice with net-30 terms. Invoices under $500 per month are
not issued; those amounts roll over to the next billing cycle.

## Are development environments billed?

Development environments are billed at half the standard CU rate ($0.02
per CU) and, because they scale to zero overnight, typically cost about
60 percent less than an always-on equivalent.

## Is there a free tier?

Yes. Every team receives 500 free compute units per month. Free units are
applied automatically to the earliest usage in the billing cycle and do
not roll over.

## How do refunds work?

Billing disputes must be raised within 45 days of the invoice date by
filing a BILLING ticket in Jira. Approved refunds are issued as platform
credit, not cash. Outages covered by the service level objective (99.9
percent monthly availability for production) generate automatic credits:
10 percent of the month's bill for availability between 99.0 and 99.9
percent, and 25 percent below 99.0 percent.

## Who can see my team's bill?

Billing reports are visible to team leads and their reporting chain in the
Aurora console under Settings > Billing. Finance partners get a monthly
export on the third business day of each month.
